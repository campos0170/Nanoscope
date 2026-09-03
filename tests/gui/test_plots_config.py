"""The Plots window's state as ONE config value — owner item A1.

His words: *"Refactor plot state into a single config object (data selection, plot type,
styling, fit params) that's fully replaced -- not patched -- on each render, to prevent
stale settings (like titles) from persisting."*

The complaint is not about tidiness. The window had EIGHT measured stale-state bugs at
once, and every one of them was the same shape: a rule about what a change invalidates,
written at the call site that made the change, and therefore missing from the call site
next to it. The axis-range boxes were hardened against a Time-unit switch and the
Fit-range boxes — two rows above them, read in the same display unit — were not. A plain
Refresh destroyed the pinned axis range and kept the typed title; a plot-type change did
the exact opposite. Legend decisions about sources and cluster rows that no longer existed
were never dropped. ``reset_state`` blanked a hand-kept list of widgets and the list had
gone stale.

So this file is in two halves.

* **The rule, as a value.** :meth:`PlotConfig.rebased` is a pure function of two
  configs' analysis identities, so most of the rule is tested without a window at all —
  which is what makes it cheap to keep exhaustive.
* **The rule, wired.** Each of the six bugs the owner named, reproduced through the real
  widgets, so that a future change that re-opens one of them fails here rather than on
  the owner's data.

Plus the contract the next round builds on: undo/redo, plot templates and JSON all need
``config()``/``_set_config`` to round-trip and need DERIVED state to stay out of the
config, so both are pinned here.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
from dataclasses import fields, replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import (
    AnalysisIdentity,
    AxesConfig,
    FitConfig,
    HistogramConfig,
    PlotConfig,
    PlotsWindow,
    ScatterConfig,
    StyleSettings,
)


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 300, *, seed: int = 3) -> pd.DataFrame:
    """A per-event frame whose dwell is exponential with tau ≈ 2.94e-4 s.

    The same shape as the owner's own −500 mV run, and the numbers the Fit-range bug was
    measured on: a range of [0.1, 2.0] ms fits ~298 of these; switching the Time unit to
    µs used to leave the boxes reading 0.1/2.0 and turn that into [0.1, 2.0] µs, a window
    a thousand times too narrow, so the fit refused with nothing on screen changed.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str = "doc0:0", label: str = "apo −500 mV", **kw) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                      colour="#0072B2", doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0), **kw)


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source()])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_time_unit("ms")
    return w


# ================================================================================================
# THE RULE, AS A VALUE — no window, no widgets, no draw
# ================================================================================================

def _hist_cfg(column: str = "dwell_s", unit: str = "ms", **axes) -> PlotConfig:
    return PlotConfig(histogram=HistogramConfig(column=column),
                      axes=AxesConfig(time_unit=unit, **axes))


def test_identity_is_the_quantities_and_nothing_else():
    """Presentation must not read as an analysis change, or every colour click would
    unpin the axes."""
    base = _hist_cfg()
    for cosmetic in (replace(base, style=StyleSettings(data_color="#ff0000")),
                     replace(base, histogram=replace(base.histogram, bins=99, log_x=True)),
                     replace(base, norm_user_set=True)):
        assert cosmetic.identity == base.identity, cosmetic

    assert _hist_cfg(column="frac_depth").identity != base.identity
    assert _hist_cfg(unit="µs").identity != base.identity
    assert replace(base, plot_type="Scatter").identity != base.identity


def test_a_time_unit_change_carries_every_typed_window_into_the_new_unit():
    """THE 1000x HAZARD, both boxes. The axis-range boxes were already carried; the
    Fit-range boxes, read in the SAME display unit, were not."""
    before = _hist_cfg(unit="ms", x_min="0.1", x_max="2.0", manual_x=(1e-4, 2e-3))
    before = replace(before, fit=FitConfig(range_lo="0.1", range_hi="2.0", range_unit="ms"))

    after = replace(before, axes=replace(before.axes, time_unit="µs")).rebased(before.identity)

    assert (float(after.axes.x_min), float(after.axes.x_max)) == pytest.approx((100.0, 2000.0))
    assert (float(after.fit.range_lo), float(after.fit.range_hi)) == pytest.approx((100.0, 2000.0))
    assert after.fit.range_unit == "µs", "the bounds must travel WITH the unit they mean"
    # The pin is in data units: the text over it moved, the window it describes did not.
    assert after.axes.manual_x == pytest.approx((1e-4, 2e-3))


def test_a_time_unit_change_leaves_a_non_time_column_alone():
    """A depth axis does not follow the Time-unit combo, so its typed limits must not
    be multiplied by 1000 when that combo moves."""
    before = _hist_cfg(column="depth_pA", unit="ms", x_min="10", x_max="80")
    before = replace(before, fit=FitConfig(range_lo="10", range_hi="80", range_unit="ms"))
    after = replace(before, axes=replace(before.axes, time_unit="µs")).rebased(before.identity)
    assert (after.axes.x_min, after.axes.x_max) == ("10", "80")
    assert (after.fit.range_lo, after.fit.range_hi) == ("10", "80")


def test_an_axis_column_change_drops_that_axis_text_and_its_pin_together():
    """0.1 ms and 0.1 of a dimensionless fraction are not the same window in different
    clothes, so there is nothing to convert — the pin goes with the text."""
    before = _hist_cfg(x_min="0.1", x_max="2.0", manual_x=(1e-4, 2e-3),
                       y_min="0", y_max="50", manual_y=(0.0, 50.0))
    after = replace(before, histogram=replace(before.histogram,
                                              column="frac_depth")).rebased(before.identity)
    assert (after.axes.x_min, after.axes.x_max) == ("", "")
    assert after.axes.manual_x is None
    # The y axis (a count) never moved, so it is untouched.
    assert (after.axes.y_min, after.axes.y_max) == ("0", "50")
    assert after.axes.manual_y == (0.0, 50.0)


def test_only_the_axis_that_moved_is_cleared():
    """A y-column change must not throw away an x pin that still means what it said."""
    before = PlotConfig(plot_type="Scatter",
                        scatter=ScatterConfig(x="dwell_s", y="depth_pA"),
                        axes=AxesConfig(x_min="0.1", manual_x=(1e-4, 2e-3),
                                        y_min="10", manual_y=(10.0, 80.0)))
    after = replace(before, scatter=replace(before.scatter,
                                            y="frac_depth")).rebased(before.identity)
    assert after.axes.x_min == "0.1" and after.axes.manual_x == (1e-4, 2e-3)
    assert after.axes.y_min == "" and after.axes.manual_y is None


def test_a_plot_type_change_unpins_both_axes_even_when_a_column_name_survives():
    """Histogram(dwell_s) -> Scatter(dwell_s, depth_pA) keeps the NAME on x and still
    changes what the pair of axes is: y stopped being a count."""
    before = _hist_cfg(x_min="0.1", manual_x=(1e-4, 2e-3), y_min="0", manual_y=(0.0, 50.0))
    after = replace(before, plot_type="Scatter",
                    scatter=ScatterConfig(x="dwell_s", y="depth_pA")).rebased(before.identity)
    assert after.axes.manual_x is None and after.axes.manual_y is None
    assert (after.axes.x_min, after.axes.y_min) == ("", "")


def test_the_fit_window_is_dropped_when_the_fitted_column_moves():
    """Bounds typed as [0.1, 2.0] for dwell silently restricted a frac_depth fit over a
    column living in [0.15, 0.45] — a fit of a handful of events, reported as a fit."""
    before = replace(_hist_cfg(), fit=FitConfig(range_lo="0.1", range_hi="2.0",
                                                range_unit="ms"))
    after = replace(before, histogram=replace(before.histogram,
                                              column="frac_depth")).rebased(before.identity)
    assert (after.fit.range_lo, after.fit.range_hi) == ("", "")


def test_the_fit_window_survives_a_plot_type_change_that_leaves_its_column_alone():
    """The Fit-range boxes bound the HISTOGRAM's column whatever page is on screen, so
    walking to the Scatter page and back must not throw the modelling decision away."""
    before = replace(_hist_cfg(), fit=FitConfig(range_lo="0.1", range_hi="2.0",
                                                range_unit="ms"))
    after = replace(before, plot_type="Scatter",
                    scatter=ScatterConfig(x="dwell_s", y="depth_pA")).rebased(before.identity)
    assert (after.fit.range_lo, after.fit.range_hi) == ("0.1", "2.0")


def test_the_caption_goes_when_the_analysis_goes_and_stays_when_it_does_not():
    """Owner note Q6: "the title for a dwell time histogram is not the same as the title
    for a frac_io vs. time KDE plot" — but a unit is not an analysis."""
    styled = StyleSettings(title="figure 3b", x_label="Dwell time (ms)", y_label="Count")
    before = replace(_hist_cfg(), style=styled)

    changed = replace(before, histogram=replace(before.histogram,
                                                column="frac_depth")).rebased(before.identity)
    assert (changed.style.title, changed.style.x_label, changed.style.y_label) == ("", "", "")

    same_quantity = replace(before,
                            axes=replace(before.axes, time_unit="µs")).rebased(before.identity)
    assert same_quantity.style.title == "figure 3b", (
        "a unit switch renames the axis, it does not change what the plot is of")


def test_an_unusable_bound_is_left_verbatim_for_the_user_to_be_told_about():
    """"nan"/"inf"/"abc" are declined by `_fit_range` and QUOTED back to the user, so the
    retune must not turn them into arithmetic."""
    before = replace(_hist_cfg(), fit=FitConfig(range_lo="nan", range_hi="abc",
                                                range_unit="ms"))
    after = replace(before, axes=replace(before.axes, time_unit="µs")).rebased(before.identity)
    assert (after.fit.range_lo, after.fit.range_hi) == ("nan", "abc")


def test_rebasing_onto_the_same_identity_is_the_identity_function():
    """Cheap, and load-bearing: `_rebase` runs on every one of these handlers, and a
    no-op that still rewrote the config would blank captions on every redraw."""
    cfg = replace(_hist_cfg(x_min="0.1", manual_x=(1e-4, 2e-3)),
                  style=StyleSettings(title="figure 3b"))
    assert cfg.rebased(cfg.identity) is cfg


def test_legend_memory_is_pruned_by_key_for_sources_and_by_offer_for_clusters():
    """The two kinds are stable against different things, so they are pruned against
    different things: a source key survives unticking and a Refresh, a cluster ordinal
    means nothing outside the run that produced it."""
    style = StyleSettings()
    for entry in ("series:doc0:0", "fit:doc0:0", "series:closed:0", "cluster:0", "cluster:3"):
        style = style.with_legend_include(entry, False)
    style = style.with_label_override("doc0:0", "apo").with_label_override("closed:0", "gone")
    cfg = PlotConfig(style=style)

    pruned = cfg.with_pruned_legend(["doc0:0"], ["series:doc0:0", "fit:doc0:0", "cluster:0"])

    assert set(pruned.style.legend_include) == {"series:doc0:0", "fit:doc0:0", "cluster:0"}
    assert dict(pruned.style.label_overrides) == {"doc0:0": "apo"}
    # An unticked-but-still-loaded source keeps its decisions: it was not offered by the
    # last draw, and pruning on that would lose the setting the moment you untick a box.
    kept = cfg.with_pruned_legend(["doc0:0", "closed:0"], [])
    assert "series:doc0:0" in kept.style.legend_include


# ================================================================================================
# THE CONTRACT the next round builds on — round-trip, value semantics, derived state out
# ================================================================================================

def test_the_config_round_trips_through_the_widgets(win):
    """``_set_config`` is the inverse of ``config()``. Undo/redo is exactly this claim
    applied twice, so it is pinned before anything is built on it."""
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_scatter_log(True, False)
    win.set_bins(44)
    win.set_title("figure 3b")
    win.set_fit("exponential")
    win._fit_lo.setText("0.1")
    win.set_axis_range(xmin=1e-4, xmax=2e-3)
    win.set_feature_selection(["dwell_s", "depth_pA"])
    taken = win.config()

    win.set_plot_type("Histogram")
    win.set_bins(7)
    win.set_title("something else")
    win._set_config(taken, draw=False)

    assert win.config() == taken


def test_the_writer_restores_the_name_typed_onto_a_source_row(win):
    """"Every control is written" has to include the Sources list's TEXT.

    That row is the legend's edit affordance (owner note #15), so its text is a
    projection of ``style.label_overrides`` exactly as the Title box is of
    ``style.title`` — and it was the one widget the writer left alone, which is how an
    undo of a rename reverted the legend and the canvas while the row kept the typed
    name. Both directions are pinned: writing an override renames the row, and writing a
    config with NO overrides puts the source's real label back.
    """
    src = win._sources["doc0:0"]
    row = win._source_list.item(0)
    plain = win.config()
    assert row.text() == src.label

    named = replace(plain, style=plain.style.with_label_override("doc0:0", "apo, −500 mV"))
    win._set_config(named, draw=False)

    assert win.config() == named, "and the round-trip contract is unchanged"
    assert row.text() == "apo, −500 mV" == win._legend_label(src)

    win._set_config(plain, draw=False)
    assert win.config() == plain
    assert row.text() == src.label, "no override means the recording's own name"
    # ...and the sync writes the TEXT and nothing else. The tick is `_apply_checked_keys`'
    # business, the colour is the series colour, and the tooltip deliberately keeps
    # naming the REAL source — that is what tells a user who retyped a row which
    # recording it still is.
    assert row.checkState() == Qt.Checked
    assert row.foreground().color().name().lower() == "#0072b2"
    assert row.toolTip().startswith(src.label)


def test_a_config_is_a_value_the_window_cannot_reach_back_into(win):
    """Held by an undo stack and by a saved template, so it has to describe the moment it
    was taken — the reason `StyleSettings` was frozen in the first place, one level up."""
    win.set_title("figure 3b")
    win.set_bins(44)
    snapshot = win.config()

    win.set_title("figure 4a")
    win.set_bins(99)
    win.set_plot_type("KDE (density)")
    win._style = win._style.with_legend_include("series:doc0:0", False)

    assert snapshot.style.title == "figure 3b"
    assert snapshot.histogram.bins == 44
    assert snapshot.plot_type == "Histogram"
    assert dict(snapshot.style.legend_include) == {}


def test_no_derived_state_ever_enters_the_config():
    """A config restored from a project file must not be able to claim a fit of rows the
    window no longer has. Named explicitly so adding one of these is a red test, not a
    review comment."""
    banned = {
        "sources", "frames", "cluster_result", "pca_result", "fits", "fit_columns",
        "last_fit", "note_cache", "pending_notes", "legend_offered", "range_factors",
        "checked_snapshot", "plot_armed", "stale", "stale_reason", "auto_plot",
        "updating", "hist_item", "scatter_item", "fit_item", "message_item",
    }

    seen = set()
    stack = [PlotConfig()]
    while stack:
        node = stack.pop()
        for f in fields(node):
            seen.add(f.name)
            value = getattr(node, f.name)
            if hasattr(type(value), "__dataclass_fields__") and not isinstance(value, tuple):
                stack.append(value)
    assert not (seen & banned), f"derived state leaked into PlotConfig: {sorted(seen & banned)}"
    assert "source_keys" in seen, (
        "the selection NAMES rows without carrying them, and that is config")


def test_the_writer_draws_at_most_once(win, monkeypatch):
    """~40 widget writes, each of which emits a change signal. Without the guard this was
    one full render per control."""
    draws = []
    monkeypatch.setattr(win, "_redraw_current", lambda: draws.append(1))
    cfg = replace(win.config(), histogram=replace(win.config().histogram, bins=44, log_x=True),
                  style=StyleSettings(title="figure 3b", point_size=9))
    win.apply_config(cfg)
    assert len(draws) == 1, f"{len(draws)} renders for one config write"

    draws.clear()
    win._set_config(replace(cfg, histogram=replace(cfg.histogram, bins=12)), draw=False)
    assert draws == [], "draw=False means the caller owns the draw"


# ================================================================================================
# THE SIX BUGS, THROUGH THE REAL WIDGETS
# ================================================================================================

def test_bug_the_fit_range_boxes_follow_a_time_unit_change(win):
    """MEASURED: fit range [0.1, 2.0] ms fits ~298 events at tau = 2.94e-4 s; switching
    the combo to µs made the same fit REFUSE, with the boxes unchanged and nothing on
    screen saying the numbers had changed meaning. The axis-range boxes were hardened
    against this ("THE 1000x HAZARD"); these were left out."""
    win._fit_lo.setText("0.1")
    win._fit_hi.setText("2.0")
    win._changed()
    in_ms = win._fit_range("dwell_s")
    assert in_ms == pytest.approx((1e-4, 2e-3))

    win.set_time_unit("µs")

    assert (float(win._fit_lo.text()), float(win._fit_hi.text())) == pytest.approx((100.0, 2000.0))
    assert win._fit_range("dwell_s") == pytest.approx(in_ms), (
        "the same physical window, in the unit now on the axis")


def test_bug_the_fit_range_boxes_do_not_survive_an_axis_column_change(win):
    """MEASURED: bounds typed for dwell still read 0.1/2.0 over a frac_depth column
    living in [0.15, 0.45], silently restricting that fit to a sliver."""
    win._fit_lo.setText("0.1")
    win._fit_hi.setText("2.0")
    win._changed()

    win.set_histogram_column("frac_depth")

    assert (win._fit_lo.text(), win._fit_hi.text()) == ("", "")
    assert win._fit_range("frac_depth") == (None, None)


def test_bug_the_pinned_range_does_not_survive_an_axis_column_change(win):
    """MEASURED: the box TEXT was rescaled while the pin under it stayed, so the axis
    stayed clamped to another quantity's numbers with nothing on screen saying so."""
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("2.0")
    win._apply_axis_range()
    assert win._manual_x == pytest.approx((1e-4, 2e-3))

    win.set_histogram_column("frac_depth")

    assert win._manual_x is None
    assert (win._range_edits["xmin"].text(), win._range_edits["xmax"].text()) == ("", "")


def test_bug_a_plain_refresh_keeps_the_pin_as_well_as_the_title(win):
    """The mirror-image pair, and the evidence the two were cleared by different ad-hoc
    rules: Refresh destroyed the pin and kept the title, a plot-type change did the
    opposite. Re-pulling the same columns is not a change of analysis, so neither goes."""
    win.set_title("Apoferritin, −500 mV")
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("2.0")
    win._apply_axis_range()

    win.set_data([_source()])   # exactly what MainWindow's Refresh does

    assert win.style().title == "Apoferritin, −500 mV"
    assert win._manual_x == pytest.approx((1e-4, 2e-3)), (
        "a Refresh re-pulls the same quantity; the pin still describes it")
    assert win._range_edits["xmin"].text() == "0.1"


def test_bug_a_refresh_that_really_does_move_a_column_clears_both(win):
    """...and the other side of the same rule. When the refreshed source no longer
    carries the plotted column, the picker moves on its own — nobody typed anything —
    and the pin and the caption are as stale as if the user had picked."""
    win.set_title("Dwell of the −500 mV pore")
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("2.0")
    win._apply_axis_range()

    thin = _events().drop(columns=["dwell_s"])
    win.set_data([PlotSource(key="doc0:0", label="apo −500 mV", df=thin, provenance={},
                             colour="#0072B2", doc_id="doc0", channel=0)])

    assert win._hist_col.currentText() != "dwell_s"
    assert win._manual_x is None
    assert win.style().title == "", "a caption for a column that is gone is a wrong caption"


def test_bug_a_stranded_cluster_exclusion_does_not_survive_a_re_clustering(win):
    """A row unticked for ``cluster:3`` was stranded across a re-clustering at a smaller
    k — silently hiding a row of a DIFFERENT clustering that happened to reuse the id.

    Only a draw that actually OFFERED a clustering is evidence about clustering, which is
    why the plain histogram redraw in the middle here must leave the exclusion standing.
    """
    win.set_feature_selection(["dwell_s", "depth_pA"])
    win.set_cluster_method("k-means")
    win.set_cluster_k(4)
    assert win.run_clustering() is not None
    assert any(i.startswith("cluster:") for i in win._legend_offered), win._legend_offered

    win._style = win._style.with_legend_include("cluster:3", False)
    assert win.style().legend_include.get("cluster:3") is False

    # A view with no clustering on it says nothing about clusters, so nothing is dropped.
    win.set_histogram_colour_by("None")
    win._redraw()
    assert "cluster:3" in win.style().legend_include, (
        "looking at another view must not discard a decision")

    win.set_histogram_colour_by("Cluster")
    win.set_cluster_k(2)
    assert win.run_clustering() is not None

    assert "cluster:3" not in win.style().legend_include, (
        "k = 2 offers no cluster 3; the exclusion was about a clustering that is gone")


def test_bug_legend_decisions_about_a_closed_source_are_dropped(win):
    """``label_overrides`` and ``legend_include`` grew monotonically: a name retyped and
    a row unticked for a source that has since been closed sat in the style forever, to
    be written into every saved template. An UNTICKED source is not a closed one and
    keeps both."""
    win.set_data([_source(), _source(key="doc1:0", label="apo −600 mV")])
    win._style = (win._style
                  .with_legend_include("series:doc0:0", False)
                  .with_legend_include("series:doc1:0", False)
                  .with_label_override("doc0:0", "apo")
                  .with_label_override("doc9:9", "a source that was closed"))

    win.set_sources(["apo −500 mV"])          # doc1 merely unticked, still loaded
    win.set_data([_source()])                 # ...and now doc1 is closed

    assert "doc9:9" not in win.style().label_overrides, "a name for a source that is gone"
    assert "series:doc1:0" not in win.style().legend_include, "a closed source's row"
    assert win.style().legend_include.get("series:doc0:0") is False
    assert win.style().label_overrides.get("doc0:0") == "apo"


def test_bug_reset_state_blanks_every_page_control_not_a_hand_kept_list(win):
    """Owner note #16: "when the user opens again everything needs to be blank". The old
    hand-kept list missed the bin count, the log axes, the KDE contour count, the
    scatter/KDE log flags, colour-by, the cluster method and k — and left the
    normalization combo on "density" while clearing the flag that remembers the user
    chose it, so the automatic default could never take that combo back."""
    win.set_bins(77)
    win.set_log_x(True)
    win.set_log_y(True)
    win.set_normalization("density")
    win._norm_user_set = True
    win.set_scatter_log(True, True)
    win.set_kde_log(False, True)
    win._kde_levels.setValue(15)
    win.set_cluster_method("k-means")
    win.set_cluster_k(7)
    win.set_title("figure 3b")
    win._fit_lo.setText("0.1")

    win.reset_state()

    blank = PlotConfig()
    assert win.config() == blank, (
        "reset_state must install a DEFAULT config, not a remembered list of widgets")
    # Spelled out for the ones the old list actually missed, so a regression names itself.
    assert win._bins.value() == blank.histogram.bins
    assert not win._hist_logx.isChecked() and not win._hist_logy.isChecked()
    assert win._norm_combo.currentText() == blank.histogram.normalization
    assert win._norm_user_set is False
    assert win._kde_levels.value() == blank.kde.levels
    assert win._k_spin.value() == blank.cluster.k
    assert win._method_combo.currentText() == blank.cluster.method
    assert win._fit_lo.text() == ""


def test_reset_state_still_leaves_the_run_gate_closed(win):
    """The gate is derived, so it is cleared beside the config rather than inside it —
    a reopened window that still counted as run would restyle live over a picture the
    user has not asked for."""
    win._redraw()
    assert win._plot_armed
    win.reset_state()
    assert not win._plot_armed and not win._stale
    assert win.sources() == [] and win.cluster_result() is None


# ================================================================================================
# the identity anchor itself
# ================================================================================================

def test_the_anchor_tracks_what_the_typed_text_was_tuned_for(win):
    """Qt has no "about to change" signal, so the whole rule rests on this attribute
    being the identity the widgets' text belongs to."""
    assert win._identity == AnalysisIdentity(
        plot_type="Histogram", x_column="dwell_s", y_column="",
        fit_column="dwell_s", time_unit="ms")

    win.set_time_unit("µs")
    assert win._identity.time_unit == "µs"
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    assert win._identity.y_column == "depth_pA"
    assert win._identity.fit_column == "dwell_s", (
        "the Fit-range boxes bound the histogram's column from every page")
