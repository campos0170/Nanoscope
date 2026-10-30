"""The Plots window as a surface a figure LEAVES from — owner notes #4, #12, #13, #15, #16.

Five complaints about the same window, all of them about control rather than about
numbers, and each one has a regression here:

* **#12 "the app applies the fit as soon as the user selects it; better to have the
  user confirm."** Opening the Fit dropdown and walking down it used to fit five
  distributions, one of which is a nine-second twelve-start optimisation. A selection
  is now a proposal; ``Apply fit`` is what draws it. The programmatic setters
  (``set_fit``, ``set_fit_components``, ``set_scatter_fit``, ``VDepWindow.set_fit``)
  stay immediate — naming the model IS the confirmation.
* **#13 "once the Fit has been applied, include the numerical values on the plot."**
  The on-plot box now carries EVERY parameter, n, BIC and the numerical KS statistic
  its own tooltip promises (the verdict itself stays in Details), and the Scatter page
  — the one that owns the scatter-fit combo — draws one at all.
* **#15 "more control over styling; the legend is not editable."** A legend entry is
  retyped in place on the source list, the legend moves to any corner, its font size
  is a control rather than a constant, and the automatic ``(n = N)`` is defeatable so
  a retyped entry is not silently extended.
* **#4 "Axis range defaults to microseconds. The user needs control over this."** The
  pin is honoured on all four plot types (it was ignored on KDE), the boxes say which
  unit they are read in, and changing that unit carries the typed text with it instead
  of leaving a 1000x-wrong number standing.
* **#16 "once the window closes the results should be saved and the panels cleared."**
  Closing files the computed fits into the session under the keys the rest of the app
  reads, then blanks the window.

The window is SHOWN in the gate tests, and that is load-bearing: the Apply gate has to
tell a person's gesture from code's, and a person's gesture is one with the popup open
or the widget holding focus — neither of which a never-shown widget can have.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotsWindow, StyleSettings
from nanoscope.gui.voltage_dep_panel import VDepWindow


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 200, *, seed: int = 3) -> pd.DataFrame:
    """A per-event frame with a dwell (seconds), a depth (pA) and an accepted flag."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(3e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.3, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"Detected events": _events()})
    return w


@pytest.fixture
def shown(win, qapp):
    """The same window, on screen — so a real gesture can be made on its widgets.

    ``hasFocus()`` is False for every widget of a window that was never shown, and
    ``QComboBox.view()`` cannot be visible either, so the Apply gate would read every
    test as "code did this" and never gate anything. Showing the window is what makes
    the person's half of the gate testable at all.
    """
    win.show()
    qapp.processEvents()
    return win


def _user_picks(qapp, combo, text: str) -> None:
    """Choose ``text`` the way a PERSON does: with the dropdown open.

    ``setCurrentText`` alone is how CODE chooses, and the two are meant to behave
    differently — that is the whole of owner note #12.
    """
    combo.showPopup()
    qapp.processEvents()
    combo.setCurrentText(text)
    qapp.processEvents()
    combo.hidePopup()
    qapp.processEvents()


# ================================================================================================
# #12 — the fit waits for Apply
# ================================================================================================

def test_picking_a_histogram_fit_does_not_draw_it_until_apply(shown, qapp):
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._fit_item is None

    _user_picks(qapp, w._fit_combo, "exponential")
    assert w._fit_combo.currentText() == "exponential"
    assert w._fit_item is None, "the selection alone must not fit anything"
    assert "τ" not in w.plot_item().titleLabel.text
    # ...and the button SAYS the plot is not showing what the combo says.
    assert w._fit_apply_btn.isEnabled()
    assert "exponential" in w._fit_apply_btn.toolTip()
    assert "none" in w._fit_apply_btn.toolTip()

    w._fit_apply_btn.click()
    qapp.processEvents()
    assert w._fit_item is not None, "Apply must draw the chosen fit"
    assert "τ" in w.plot_item().titleLabel.text
    assert not w._fit_apply_btn.isEnabled(), "nothing is pending once it is applied"


def test_the_component_count_is_part_of_what_apply_commits(shown, qapp):
    """k is a property of the fit, so it waits at the same gate the family does."""
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    w.set_fit("gaussian-mixture")           # programmatic: applied at once
    w.set_fit_components(1)
    assert w._applied_fit_k == 1

    w._fit_k.setFocus()                     # a person reaching for the spin
    qapp.processEvents()
    w._fit_k.setValue(2)
    qapp.processEvents()
    assert w._applied_fit_k == 1, "a typed k must not refit before Apply"
    assert w._fit_apply_btn.isEnabled()

    w._fit_apply_btn.click()
    qapp.processEvents()
    assert w._applied_fit_k == 2
    assert w.fit_result().meta["n_components"] == 2


def test_picking_a_scatter_fit_does_not_draw_it_until_apply(shown, qapp):
    w = shown
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    assert w._scatter_fit_item is None

    _user_picks(qapp, w._scatter_fit, "linear")
    assert w._scatter_fit_item is None, "the selection alone must not fit anything"
    assert w._scatter_fit_apply_btn.isEnabled()

    w._scatter_fit_apply_btn.click()
    qapp.processEvents()
    assert w._scatter_fit_item is not None
    assert "R²" in w.details_text()
    assert not w._scatter_fit_apply_btn.isEnabled()


def test_the_programmatic_setters_stay_immediate(win):
    """The gate is about the GESTURE. A caller that names a model has already decided,
    and a setter that needed a button press afterwards would be the silent no-op
    ``set_fit``'s own docstring exists to prevent."""
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    win.set_fit("exponential")
    assert win._fit_item is not None and win._applied_fit == "exponential"

    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_scatter_fit("linear")
    assert win._scatter_fit_item is not None
    assert win._applied_scatter_fit == "linear"


def test_the_voltage_dependence_trend_waits_for_apply_too(qapp, qtbot):
    w = VDepWindow(Session())
    qtbot.addWidget(w)
    w.show()
    qapp.processEvents()
    assert w._fit_model() == "none"

    _user_picks(qapp, w._fit_combo, "linear")
    assert w._fit_combo.currentText() == "linear"
    assert w._fit_model() == "none", "the selection alone must not fit the trend"
    assert w._fit_apply_btn.isEnabled()

    w._fit_apply_btn.click()
    qapp.processEvents()
    assert w._fit_model() == "linear"
    assert not w._fit_apply_btn.isEnabled()

    # ...and the programmatic setter is itself the confirmation.
    w.set_fit("exponential")
    assert w._fit_model() == "exponential"


# ================================================================================================
# #13 — the numbers land ON the plot
# ================================================================================================

def test_the_on_plot_box_carries_every_parameter_not_just_the_headline(win):
    """A two-component mixture put its means on the figure and left both sigmas and
    both weights in a Details pane the exported PNG does not contain."""
    win.set_plot_type("Histogram")
    win.set_histogram_column("depth_pA")
    win.set_fit("gaussian-mixture")
    win.set_fit_components(2)
    win._fit_values_check.setChecked(True)

    text = win._fit_values_item.text
    assert "μ_1" in text and "μ_2" in text, text
    assert "σ_1" in text and "σ_2" in text, "the widths were the missing half"
    assert "n = " in text and "BIC = " in text, text
    # The Style checkbox's tooltip promises the numerical KS statistic; the box delivers
    # it. (The verdict stays in Details, not on the figure, since 779d4ec, 2026-09-06.)
    assert "KS = " in text, text
    assert "numerical KS statistic" in win._fit_values_check.toolTip()


def test_the_scatter_page_draws_a_fit_values_box_at_all(win):
    """The checkbox lived beside the fit colour and worked only on the histogram — so
    the page that owns the scatter-fit combo was the one page where ticking it did
    nothing."""
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win._fit_values_check.setChecked(True)
    assert win._fit_values_item is None, "no fit, no box"

    win.set_scatter_fit("linear")
    assert win._fit_values_item is not None, "the Style checkbox must reach the scatter"
    assert "R²" in win._fit_values_item.text
    assert "y = " in win._fit_values_item.text


def test_the_matplotlib_export_path_annotates_and_can_be_told_not_to():
    """The OTHER half of note #13: "make these plots presentation ready" is about the
    figure that LEAVES the app, and the app has two renderers of it.

    ``Analysis ▸ Export figures`` goes through :mod:`nanoscope.plot.figures`, where the
    two histograms wrote their numbers unconditionally (no way to turn them off for a
    multi-panel with a shared caption) and the canonical dwell-vs-depth scatter wrote
    none at all. Lives here rather than in ``tests/plot`` because it is the same owner
    note as the on-plot box above, and the two must not drift apart.
    """
    import matplotlib
    matplotlib.use("Agg")
    from nanoscope.plot import figures

    df = _events(400)
    df["frac_depth"] = np.abs(np.random.default_rng(0).normal(0.3, 0.05, len(df)))

    def _texts(ax):
        return " ".join(t.get_text() for t in ax.texts)

    fig, ax = figures.dwell_histogram(df, fit=True)
    assert "n = " in _texts(ax)
    fig, ax = figures.dwell_histogram(df, fit=True, annotate=False)
    assert "n = " not in _texts(ax)

    fig, ax = figures.depth_histogram(df, fit=True)
    assert r"$\mu$" in _texts(ax)
    fig, ax = figures.depth_histogram(df, fit=True, annotate=False)
    assert r"$\mu$" not in _texts(ax)

    # The scatter carried no numbers at all; now it carries the ones it measured.
    fig, ax = figures.scatter_dwell_depth(df, density=True)
    text = _texts(ax)
    assert "n = 400" in text and "median dwell" in text, text
    fig, ax = figures.scatter_dwell_depth(df, density=True, annotate=False)
    assert "median dwell" not in _texts(ax)


# ================================================================================================
# #15 — the legend is editable, movable and sizeable
# ================================================================================================

def _source(key: str, df: pd.DataFrame, *, colour: str, label: str) -> PlotSource:
    return PlotSource(key=key, label=label, df=df, provenance={"fs": 250_000.0},
                      colour=colour, doc_id=key, channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def two_sources(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([
        _source("a", _events(seed=1), colour="#0072B2", label="apo_0008_-500 mV.csv"),
        _source("b", _events(seed=2), colour="#D55E00", label="apo_0009_-700 mV.csv"),
    ])
    w.set_sources(["a", "b"])
    return w


def _legend_texts(w) -> list[str]:
    legend = getattr(w.plot_item(), "legend", None)
    if legend is None:
        return []
    return [label.text for _sample, label in legend.items]


class _Drag:
    """The four things ``LegendItem.mouseDragEvent`` reads off a drag event.

    Driving the REAL handler with a stand-in event rather than re-implementing what a
    drag does: pyqtgraph's handler is the code under test here (it is what makes the
    legend interactive at all), and the panel's only contribution is remembering where
    it left the item. Qt's own event delivery is the one link this skips, and it is
    the link that is not this window's to get wrong.
    """

    def __init__(self, dx: float, dy: float) -> None:
        self._d = (dx, dy)
        self.accepted = False

    def button(self):
        return Qt.LeftButton

    def accept(self) -> None:
        self.accepted = True

    def pos(self):
        from PySide6.QtCore import QPointF
        return QPointF(*self._d)

    def lastPos(self):
        from PySide6.QtCore import QPointF
        return QPointF(0.0, 0.0)


def _drag_legend(w, dx: float, dy: float):
    """Drag the legend by (dx, dy) px and return its new position."""
    legend = w.plot_item().legend
    ev = _Drag(dx, dy)
    legend.mouseDragEvent(ev)
    assert ev.accepted, "premise: pyqtgraph accepted the left-button drag"
    return legend.pos()


def _legend_fraction(w):
    """Where the legend sits now, as the (fx, fy) the style stores."""
    legend = w.plot_item().legend
    br = legend.parentItem().boundingRect()
    return (legend.pos().x() / br.width(), legend.pos().y() / br.height())


def test_a_retyped_source_row_renames_its_legend_entry(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert any("apo_0008" in t for t in _legend_texts(w))

    item = w._source_list.item(0)
    assert bool(item.flags() & Qt.ItemIsEditable), "the row must be editable in place"
    item.setText("apo, −500 mV")
    assert w._style.label_overrides["a"] == "apo, −500 mV"
    assert any("apo, −500 mV" in t for t in _legend_texts(w))
    # The source itself is untouched — the override is a display name, not a rename.
    assert w.sources()[0].label.endswith("-500 mV.csv")

    # ...and it reaches the scatter and the KDE, not only the histogram.
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    assert any("apo, −500 mV" in t for t in _legend_texts(w))
    w.set_plot_type("KDE (density)")
    w.set_kde_columns("dwell_s", "depth_pA")
    assert any("apo, −500 mV" in t for t in _legend_texts(w))

    # Clearing the edit restores the source's own label.
    w.set_plot_type("Histogram")
    w._source_list.item(0).setText("")
    assert "a" not in w._style.label_overrides
    assert w._source_list.item(0).text().endswith("-500 mV.csv")


def test_the_automatic_count_suffix_can_be_turned_off(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert all("(n = " in t for t in _legend_texts(w))

    w._legend_counts.setChecked(False)
    texts = _legend_texts(w)
    assert texts and not any("(n = " in t for t in texts), (
        "an entry the user retyped must not be silently extended with a count")


def test_a_fresh_legend_starts_in_the_default_corner(two_sources):
    """A plot nobody has touched still puts its legend where it always did."""
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    legend = w.plot_item().legend
    assert w._style.legend_pos is None, "nothing has been dragged"
    assert tuple(legend.opts["offset"]) == pp._LEGEND_DEFAULT_OFFSET


def test_the_four_position_combo_is_gone(win):
    """Round-4 Q4: "the location should not be limited to four fixed positions in the
    panel — it should be interactive". The combo is the thing being rejected, and its
    per-draw ``setOffset`` is what made the legend undraggable, so it is removed rather
    than merely ignored — a control that fights the drag is worse than no control."""
    assert not hasattr(win, "_legend_loc")
    assert not hasattr(pp, "_LEGEND_OFFSETS")
    assert not hasattr(pp.StyleSettings(), "legend_loc")


# ================================================================================================
# Round-4 Q4 — the legend is placed by DRAGGING it, and the drag is remembered
#
# This is a regression, not a feature: pyqtgraph 0.14's LegendItem has always been
# draggable, and the four-corner combo's per-draw ``setOffset`` is what threw the drag
# away. Measured before the fix, on this same two-source overlay: a drag moved the
# legend from (487.0, 12.0) to (287.0, 192.0) and the very next redraw put it back at
# (486.7, 12.0). Every test below is that measurement, kept.
# ================================================================================================

def test_dragging_the_legend_moves_it_and_is_remembered(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    before = w.plot_item().legend.pos()
    assert w._style.legend_pos is None

    after = _drag_legend(w, -200, 180)
    assert (after.x(), after.y()) != (before.x(), before.y()), "the drag did nothing"
    assert after.x() == pytest.approx(before.x() - 200)
    assert after.y() == pytest.approx(before.y() + 180)
    # ...and the panel read it back off the item, as a fraction of the plot area.
    assert w._style.legend_pos is not None
    assert w._style.legend_pos == pytest.approx(_legend_fraction(w))


def test_the_dragged_legend_survives_a_redraw(two_sources):
    """THE MEASURED REGRESSION. ``_ensure_legend`` called ``setOffset`` on every draw,
    which re-anchors to a corner, so the next redraw discarded the user's drag."""
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    dropped = _drag_legend(w, -200, 180)

    w.set_bins(17)                        # any redraw at all
    assert w.plot_item().legend.pos() == dropped

    w._font_size.setValue(14)             # ...including a live restyle
    assert w.plot_item().legend.pos() == dropped


def test_the_dragged_legend_survives_a_plot_type_change(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    _drag_legend(w, -200, 180)
    want = w._style.legend_pos

    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    assert _legend_fraction(w) == pytest.approx(want)

    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert _legend_fraction(w) == pytest.approx(want)


def test_the_dragged_legend_survives_an_image_export(two_sources, tmp_path):
    """The export is the point of moving it at all, so the claim is made against the
    PIXELS: the same plot saved with the legend in the corner and with the legend
    dragged has to be a different image."""
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    corner = w.save_image(str(tmp_path / "corner.png"))
    assert corner is not None and corner.stat().st_size > 0

    _drag_legend(w, -200, 180)
    want = w._style.legend_pos
    moved = w.save_image(str(tmp_path / "moved.png"))
    assert moved is not None and moved.stat().st_size > 0
    assert moved.read_bytes() != corner.read_bytes(), (
        "the exported figure did not follow the legend")
    # ...and exporting did not itself disturb the position.
    assert w._style.legend_pos == pytest.approx(want)
    assert _legend_fraction(w) == pytest.approx(want)


def test_reset_puts_the_legend_back_in_the_corner(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    home = w.plot_item().legend.pos()
    assert not w._legend_reset_btn.isEnabled(), "nothing to reset on a fresh plot"

    _drag_legend(w, -200, 180)
    assert w._legend_reset_btn.isEnabled()

    w._legend_reset_btn.click()
    assert w._style.legend_pos is None
    assert w.plot_item().legend.pos() == home
    assert not w._legend_reset_btn.isEnabled()


def test_re_applying_a_remembered_position_does_not_creep(two_sources):
    """``autoAnchor`` is idempotent and the panel leans on that, so the claim is
    measured: five redraws in a row must not walk the legend across the plot."""
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    dropped = _drag_legend(w, -200, 180)
    for i in range(5):
        w.set_bins(20 + i)
        assert w.plot_item().legend.pos() == dropped, f"drifted on redraw {i + 1}"


# ================================================================================================
# Round-4 Q2 — WHICH legend entries
# ================================================================================================

def test_a_fitted_histogram_offers_both_rows_and_either_can_be_dropped(win):
    """The owner's complaint verbatim: "when the fit is performed and the legend is
    selected, the app includes icons for both the histogram and the lines. This is
    incorrect; the user needs control over which icons to add to the legend"."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")

    ids = list(w._legend_offered)
    assert any(i.startswith("series:") for i in ids), ids
    assert any(i.startswith("fit:") for i in ids), ids
    assert len(_legend_texts(w)) == len(ids), "both rows are drawn by default"

    fit_id = next(i for i in ids if i.startswith("fit:"))
    w._style = w._style.with_legend_include(fit_id, False)
    w._changed()
    texts = _legend_texts(w)
    assert texts and not any("fit:" in t for t in texts), texts

    # ...and the other way round: keep the fit's tau, drop the bars.
    w._style = replace(
        w._style,
        legend_include=pp.FrozenMap(
            {next(i for i in ids if i.startswith("series:")): False}))
    w._changed()
    texts = _legend_texts(w)
    assert len(texts) == 1 and "fit:" in texts[0], texts


def test_the_legend_entry_checklist_lists_what_the_draw_offered(win):
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    lst = w._legend_entries
    assert lst.count() == len(w._legend_offered) >= 2
    assert all(lst.item(i).checkState() == Qt.Checked for i in range(lst.count()))

    # Unticking a row is what drops it, and the row STAYS listed so it can come back.
    row = next(i for i in range(lst.count())
               if str(lst.item(i).data(Qt.UserRole)).startswith("fit:"))
    before = len(_legend_texts(w))
    lst.item(row).setCheckState(Qt.Unchecked)
    assert len(_legend_texts(w)) == before - 1
    assert w._legend_entries.count() == lst.count(), "the unticked row is still offerable"

    w._legend_entries.item(row).setCheckState(Qt.Checked)
    assert len(_legend_texts(w)) == before


def test_legend_inclusion_is_keyed_per_source_not_per_label(two_sources):
    """Two overlaid pores are two rows, and dropping one must not drop the other —
    which is why the id is the source KEY and not the (retypable) label."""
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert len(_legend_texts(w)) == 2

    w._style = w._style.with_legend_include("series:a", False)
    w._changed()
    texts = _legend_texts(w)
    assert len(texts) == 1 and "0009" in texts[0], texts

    # Retyping the surviving entry does not resurrect or lose the hidden one.
    w._source_list.item(1).setText("−700 mV")
    texts = _legend_texts(w)
    assert texts == ["−700 mV  (n = 200)"], texts


# ================================================================================================
# Round-4 Q3 — the legend text is editable, and now it SAYS so
# ================================================================================================

def test_the_source_list_advertises_that_its_rows_are_editable(two_sources):
    """The feature worked; nothing told the user it was there."""
    w = two_sources
    box = next(g for g in w.findChildren(QtWidgets.QGroupBox) if g.title() == "Data sources")
    hints = " ".join(lbl.text() for lbl in box.findChildren(QtWidgets.QLabel))
    assert "Double-click" in hints and "legend" in hints.lower()
    assert "✎" in hints, "the hint has to read as an edit at a glance, not only in prose"

    assert "Double-click" in w._source_list.toolTip()
    assert "Double-click" in w._source_list.item(0).toolTip()
    # ...and the rows really are editable, which is what the hint is promising.
    assert w._source_list.item(0).flags() & Qt.ItemIsEditable


# ================================================================================================
# Round-4 Q5 / Q7 — independent text sizes, and a grid the user owns
# ================================================================================================

def test_one_font_size_no_longer_drives_every_text(win):
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    # A plain histogram has no automatic title, so one is typed — the title has to
    # EXIST before a claim about its size means anything. (Typed after the column, so
    # the Q6 clear-on-analysis-change rule does not take it away again.)
    w.set_title("Figure 3b")
    # The single-knob behaviour is the DEFAULT: every override starts at "follow body".
    w.set_font_size(12)
    assert (w._title_pt(), w._axis_label_pt(), w._tick_pt()) == (12, 12, 12)
    assert "12pt" in w.plot_item().titleLabel.opts["size"]

    w._title_size.setValue(26)
    w._axis_label_size.setValue(18)
    w._tick_size.setValue(7)
    assert (w._title_pt(), w._axis_label_pt(), w._tick_pt()) == (26, 18, 7)
    assert "26pt" in w.plot_item().titleLabel.opts["size"]
    assert w.plot_item().getAxis("bottom").style["tickFont"].pointSize() == 7
    # ...and the body size no longer drags them with it.
    w.set_font_size(9)
    assert (w._title_pt(), w._axis_label_pt(), w._tick_pt()) == (26, 18, 7)

    # 0 means "follow the body again", which is how a user gets back.
    w._title_size.setValue(0)
    assert w._title_pt() == 9


def test_the_title_can_be_bold_or_italic_without_choosing_a_font_family(win):
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("Figure 3b")
    assert w._style.font_family == "", "premise: no family has been chosen"
    assert not w.plot_item().titleLabel.item.font().bold()

    w._title_bold.setChecked(True)
    w._title_italic.setChecked(True)
    font = w.plot_item().titleLabel.item.font()
    assert font.bold() and font.italic()


def test_gridlines_are_a_control_and_are_re_read_on_every_draw(win):
    """Measured before: ``showGrid`` ran once at construction and was never read
    again, so this was the one part of the canvas the Style group could not touch."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._style.show_grid is True
    assert w.plot_item().ctrl.xGridCheck.isChecked()

    w._grid_check.setChecked(False)
    assert w._style.show_grid is False
    assert not w.plot_item().ctrl.xGridCheck.isChecked()
    assert not w.plot_item().ctrl.yGridCheck.isChecked()

    # ...and it survives the next draw, which is the half the constructor call could
    # never do.
    w.set_histogram_column("depth_pA")
    assert not w.plot_item().ctrl.xGridCheck.isChecked()

    w._grid_check.setChecked(True)
    assert w.plot_item().ctrl.xGridCheck.isChecked()


def test_the_legend_font_size_is_a_control_with_the_old_rule_as_its_default(two_sources):
    w = two_sources
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    # 0 = "follow the body font, three points below it" — today's hard-coded rule.
    assert w._style.legend_font_size_pt == 0
    w.set_font_size(20)
    assert w.plot_item().legend.opts["labelTextSize"] == "17pt"

    w._legend_font_size.setValue(9)
    assert w.plot_item().legend.opts["labelTextSize"] == "9pt"


# ================================================================================================
# #4 / #18 — the axis pin holds on every plot type, in a unit the boxes name
# ================================================================================================

@pytest.mark.parametrize("kind, setup", [
    ("Histogram", lambda w: w.set_histogram_column("dwell_s")),
    ("Scatter", lambda w: w.set_scatter_columns("dwell_s", "depth_pA")),
    ("KDE (density)", lambda w: (w.set_kde_columns("dwell_s", "depth_pA"),
                                 w.set_kde_log(False, False))),
    ("PCA", lambda w: None),
])
def test_a_pinned_axis_range_survives_a_redraw_on_every_plot_type(win, kind, setup):
    """KDE ranged its view on the grid bounds UNCONDITIONALLY, so the pin — the whole
    Axis-range group — was ignored on that one page."""
    win.set_plot_type(kind)
    setup(win)
    if kind == "PCA":
        win.set_feature_selection(["dwell_s", "depth_pA", "snr"])
        win.run_pca()

    win.set_axis_range(xmin=1e-4, xmax=8e-4, ymin=1.0, ymax=60.0)
    pinned = [float(v) for r in win.plot_item().vb.viewRange() for v in r]
    assert pinned == pytest.approx([1e-4, 8e-4, 1.0, 60.0]), (
        f"the pin was not applied at all on the {kind} page")
    win._redraw()
    after = [float(v) for r in win.plot_item().vb.viewRange() for v in r]
    assert after == pytest.approx(pinned), (
        f"the pinned range did not survive a redraw of the {kind} page")


def test_the_range_boxes_say_which_unit_they_are_read_in(win):
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    assert win._range_labels["xmin"].text() == "X min (ms)"
    assert win._range_labels["ymin"].text() == "Y min", "a count has no unit"

    win.set_time_unit("µs")
    assert win._range_labels["xmax"].text() == "X max (µs)"

    # A non-time column takes its unit from the schema that declares it.
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    assert win._range_labels["xmin"].text() == "X min (µs)"
    assert win._range_labels["ymin"].text() == "Y min (pA)"


def test_changing_the_time_unit_carries_the_typed_limits_with_it(win):
    """THE 1000x HAZARD. The boxes are read in the DISPLAY unit; switching ms to µs
    left "0.5" standing and the next Apply read it as a window 1000x too narrow."""
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    win.set_time_unit("ms")
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("0.5")
    win._apply_axis_range()
    assert win._manual_x == pytest.approx((1e-4, 5e-4))

    win.set_time_unit("µs")
    assert float(win._range_edits["xmin"].text()) == pytest.approx(100.0)
    assert float(win._range_edits["xmax"].text()) == pytest.approx(500.0)
    # The pin itself never moved: only the text over it changed unit.
    win._apply_axis_range()
    assert win._manual_x == pytest.approx((1e-4, 5e-4))


def test_switching_to_a_picoamp_axis_DROPS_a_dwell_pin(win):
    """A pin typed for a dwell axis is not a pin on a picoamp one — it is dropped.

    EXPECTATION CHANGED, DELIBERATELY (item A1, the third stale-state bug). This test
    used to assert the opposite half of the same operation: that the box TEXT was
    rescaled (0.1 ms -> 1e-4) while ``_manual_x`` stayed pinned at (1e-4, 5e-4), on the
    grounds that "the stored pin is in data units and must not move". Both halves were
    wrong together, and they were wrong in a way that only shows on screen: the view
    stayed clamped to [1e-4, 5e-4] — a window 0.0005 pA wide over a depth axis living
    near 40 pA, so the plot came up empty — while the boxes displayed a rescaled number
    that made it look deliberate. The conversion is only meaningful between two SPELLINGS
    of one quantity (0.1 ms and 100 µs are the same window); between two quantities there
    is nothing to convert, because 0.1 ms is not a picoamp count at all.

    So the rule (``PlotConfig.rebased``, clause 2) drops the text and the pin TOGETHER
    when the column on an axis changes. The Time-unit case above keeps both, unchanged,
    which is the distinction: a unit is presentation, a column is the analysis.
    """
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_time_unit("ms")
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("0.5")
    win._apply_axis_range()
    assert win._manual_x == pytest.approx((1e-4, 5e-4))

    win.set_scatter_columns("depth_pA", "depth_pA")   # x is no longer a time column
    assert win._range_labels["xmin"].text() == "X min (pA)"
    assert win._range_edits["xmin"].text() == "" and win._range_edits["xmax"].text() == ""
    assert win._manual_x is None, "a dwell pin must not survive onto a picoamp axis"
    # The Y axis never moved off depth_pA, so nothing about it is disturbed.
    assert win._manual_y is None and win._range_edits["ymin"].text() == ""


# ================================================================================================
# #16 — close saves the results, then blanks the window
# ================================================================================================

def _detected_source(session: Session) -> PlotSource:
    return PlotSource(key="doc0:0", label="apo −500 mV", df=_events(),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                      colour="#0072B2", doc_id="doc0", channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


def test_closing_files_the_computed_fit_into_the_session(qapp, qtbot):
    """The window was read-only against the session, so a fit the user set up here —
    family, k, fit range, KS verdict — was thrown away on close while the identical
    fit made in the Fits panel survived."""
    session = Session()
    w = PlotsWindow(session)
    qtbot.addWidget(w)
    w.set_data([_detected_source(session)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    assert w.fit_result() is not None
    assert session.results.get("doc0", 0, "fit.dwell") is None

    w.close()
    stored = session.results.get("doc0", 0, "fit.dwell")
    assert stored is not None, "the fit went nowhere when the window closed"
    assert hasattr(stored.payload, "predict"), "the payload must be the FitResult itself"
    assert stored.params["column"] == "dwell_s"
    assert tuple(stored.calibration_fingerprint) == (1.0, 0.0)
    assert stored.scalars["bic"] == pytest.approx(float(stored.payload.bic))


def test_a_depth_fit_is_filed_under_the_depth_key(qapp, qtbot):
    session = Session()
    w = PlotsWindow(session)
    qtbot.addWidget(w)
    w.set_data([_detected_source(session)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    w.set_fit("gaussian-mixture")
    w.close()
    assert session.results.get("doc0", 0, "fit.depth") is not None
    assert session.results.get("doc0", 0, "fit.dwell") is None


def test_a_reopened_window_is_blank_and_carries_no_style_or_pin(qapp, qtbot):
    """"When the user opens again everything needs to be blank." The window is built
    once and re-shown, so without a reset it came back carrying the last session's
    title, its pinned range, its light background and its renamed legend entries."""
    session = Session()
    w = PlotsWindow(session)
    qtbot.addWidget(w)
    w.set_data([_detected_source(session)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    w.set_title("Figure 3b")
    w.set_font_size(18)
    w.set_time_unit("µs")
    w._bg_combo.setCurrentText("Light")
    _drag_legend(w, -120, 90)
    w._source_list.item(0).setText("apo")
    w.set_axis_range(xmin=1e-4, xmax=8e-4)
    assert w._manual_x is not None

    w.close()

    defaults = StyleSettings()
    assert w._sources == {} and w._fits == {}
    assert w.details_text() == "" or "No data yet" in w.details_text()
    assert w._manual_x is None and w._manual_y is None
    assert all(not e.text() for e in w._range_edits.values())
    assert w._style.title == defaults.title
    assert w._style.font_size_pt == defaults.font_size_pt
    assert w._style.background == defaults.background
    assert w._style.legend_pos == defaults.legend_pos
    assert w._style.label_overrides == {}
    assert w._applied_fit == "none" and w._applied_scatter_fit == "none"
    assert w.time_unit() == pp._DEFAULT_TIME_UNIT
    assert w._plot_type_combo.currentText() == pp.HISTOGRAM

    # Reopening (the owner re-pushes data before show()) must not resurrect any of it.
    w.set_data([_detected_source(session)])
    assert w._style.title == "" and w._style.background == "dark"
    assert w._manual_x is None and w._manual_y is None
    assert w._style.label_overrides == {}
    assert w._fit_item is None, "a reopened window must not still be fitting"
    assert "No data yet" not in w.details_text(), "the re-pushed data draws normally"


# ================================================================================================
# S6-01 — the Font row names the font the plot is DRAWN in
#
# ``StyleSettings.font_family`` defaults to "" = "whatever font this computer hands the
# application", and a ``QFontComboBox`` cannot show that: the resolved family is
# ``.AppleSystemUIFont``, which it filters out as private, so it fell back to its first
# entry and told the user their figure was in "Academy Engraved LET". The blank is a
# CHOICE, not a missing value, so it now has a row of its own that says so in words — and
# every row carries its style value in ``Qt.UserRole``, which is what makes "" a value the
# widget can finally spell. These four tests pin both halves: what the row SAYS, and that
# nothing but a real gesture ever writes a family into the style.
# ================================================================================================

def test_the_font_row_says_application_default_rather_than_naming_a_font(win, qtbot):
    w = win
    assert w._style.font_family == "", "premise: no family has been chosen"

    assert w._font_combo.currentIndex() == 0
    assert w._font_combo.currentData() == "", "the row carries the STYLE value, not prose"
    assert w._font_combo.currentText() == pp._FONT_DEFAULT_LABEL
    # THE ASSERTION THAT PINS THE DEFECT. Before the fix this read "Academy Engraved LET",
    # an installed family that exactly matches — i.e. the panel named a decorative
    # engraved face for a figure drawn in the system UI font.
    assert not QtGui.QFont(w._font_combo.currentText()).exactMatch(), (
        "the Font row must not name an installed family when none was chosen")

    # ...and the replacement offers exactly the app's families, not a set of its own.
    #
    # A FRESH WINDOW, AFTER FORCING QT'S LAZY ALIAS POPULATION. Measured: asking for a
    # family that does not exist grows ``QFontDatabase.families()`` by two alias names
    # (Noto Sans Armenian, Noto Sans Zawgyi) on BOTH platforms, and until something
    # triggers it the count differs by exactly those two — under cocoa the trigger fires
    # inside window construction (list 181, later 183), under offscreen before it. The
    # window snapshots the list at construction, so force the growth first and the
    # comparison is exact rather than order-dependent.
    QtGui.QFontInfo(QtGui.QFont("no such family — force Qt's alias population")).family()
    fresh = PlotsWindow(Session())
    qtbot.addWidget(fresh)
    families = pp._font_families()
    assert fresh._font_combo.count() == len(families) + 1
    assert [fresh._font_combo.itemText(i)
            for i in range(1, fresh._font_combo.count())] == families
    assert [fresh._font_combo.itemData(i)
            for i in range(1, fresh._font_combo.count())] == families, (
        "a family row must carry its own name as its style value")


def test_the_default_font_row_is_a_value_the_user_can_choose_back(win):
    """"Application default" is a CHOICE, so it has to be reachable in both directions —
    picked away from and picked back to, through the widget and through the setter."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")

    row = w._font_combo.findData("Courier New")
    assert row > 0, "premise: this machine has Courier New"
    w._font_combo.setCurrentIndex(row)                  # a real pick, through the widget
    assert w._style.font_family == "Courier New"
    assert w._qfont().family() == "Courier New", "the plot follows the row"

    w._font_combo.setCurrentIndex(0)                    # ...and back to the default
    assert w._style.font_family == "", "row 0 commits the blank, not the word it shows"
    assert w._qfont().family() == QtGui.QFont().family()

    # The setter half. Before S6-01 `setCurrentFont(QFont(""))` left the combo showing
    # whatever it had been showing while the style went blank — the panel and the figure
    # disagreeing, which is the defect in the other direction.
    w.set_font_family("Courier New")
    assert w._font_combo.currentText() == "Courier New"
    w.set_font_family("")
    assert w._style.font_family == ""
    assert w._font_combo.currentIndex() == 0
    assert w._font_combo.currentData() == ""


def test_a_style_sync_never_commits_a_font_the_user_did_not_choose(win):
    """The half of S6-01 that was NEVER broken, and must not become broken by the fix.

    Every one of these paths writes ~40 controls from a config, and the Font row now MOVES
    ROWS while it does it (it can insert and remove one). An unblocked move would reach
    ``_on_style_font_family`` and write a family nobody picked — so ``_select_font_family``
    blocks the combo for its own body, and this drives every caller through it. (That the
    block is a guarantee rather than a precondition the caller has to remember is pinned
    on its own, below, in
    ``test_pointing_the_font_row_at_a_family_cannot_commit_it_whatever_the_caller_did``.)"""
    w = win
    assert w._style.font_family == ""

    w.apply_config(w.config())
    assert (w._style.font_family, w._font_combo.currentIndex()) == ("", 0)

    w.apply_template(w.config().as_template())
    assert (w._style.font_family, w._font_combo.currentIndex()) == ("", 0)

    w.reset_state()
    assert (w._style.font_family, w._font_combo.currentIndex()) == ("", 0)

    # ONE GESTURE IS ONE UNDO STEP — the thing the swap from `currentFontChanged` to
    # `currentIndexChanged` could quietly have doubled.
    back, _ = w._history.depth()
    w._font_combo.setCurrentIndex(w._font_combo.findData("Courier New"))
    assert w._style.font_family == "Courier New"
    assert w._history.depth()[0] == back + 1

    assert w.undo_config()
    assert (w._style.font_family, w._font_combo.currentIndex()) == ("", 0), (
        "undo restores the blank AND the row that stands for it"
    )
    assert w.redo_config()
    assert w._style.font_family == "Courier New"
    assert w._font_combo.currentData() == "Courier New"


def test_a_family_this_machine_lacks_is_shown_as_absent_not_rewritten(win):
    """A config exported on a colleague's machine can name a font this one has not got.

    Dropping it would leave the previous row standing (the same lie in a new place) and
    matching it to the nearest family would silently rewrite the user's style, so the exact
    string gets a row of its own, marked as absent, and the style keeps it verbatim."""
    w = win
    if "calibri" in w._font_rows:
        pytest.skip("this machine has Calibri; the test needs a family it lacks")

    cfg = w.config()
    w.apply_config(replace(cfg, style=replace(cfg.style, font_family="Calibri")))

    assert w._style.font_family == "Calibri", "not rewritten to something installed"
    assert w._font_combo.currentData() == "Calibri", "the row carries the exact string"
    assert "not on this computer" in w._font_combo.currentText()
    assert w.config().style.font_family == "Calibri", "and it still exports"

    # The extra row is temporary: the next sync clears it.
    cfg = w.config()
    w.apply_config(replace(cfg, style=replace(cfg.style, font_family="")))
    assert w._font_combo.currentIndex() == 0 and w._style.font_family == ""
    assert w._font_combo.findData("Calibri") == -1, "the absent-family row is gone"

    # A HAND-SPELLED FAMILY selects the real row, because QFont matching is
    # case-insensitive and "courier new" really is drawn in Courier New — while the style
    # keeps the spelling the user (or their sidecar) wrote.
    cfg = w.config()
    w.apply_config(replace(cfg, style=replace(cfg.style, font_family="courier new")))
    assert w._font_combo.currentText() == "Courier New"
    assert w._font_combo.currentData() == "Courier New"
    assert w._style.font_family == "courier new"
    assert w.config().style.font_family == "courier new"


# ------------------------------------------------------------------------------------------------
# S6-01, round 2 — the three things the verifier found still owing on the Font row.
#
# None of them draws a wrong number; all three are the row telling the user something that
# is not so. The absent-family row outlived the pick that replaced it, the tooltip printed
# a private platform name that the list itself refuses to show, and the rule that keeps a
# sync from committing a font was an ``assert`` — which ``python -O`` deletes.
# ------------------------------------------------------------------------------------------------

def test_the_absent_family_row_is_retired_by_the_choice_that_replaces_it(win):
    """"Calibri — not on this computer" is about a decision, so it goes when the decision does.

    A config from a colleague's machine names a family this one has not got and gets a row
    of its own saying so. Before this fix that row was cleared only by the NEXT
    ``_select_font_family`` — an undo, an import, a reset — and NOT by the user picking a
    different font, which is the very gesture that answers it. So one imported config plus
    one ordinary font change left the dropdown offering a font this computer cannot draw,
    for the rest of the session.
    """
    w = win
    if "calibri" in w._font_rows:
        pytest.skip("this machine has Calibri; the test needs a family it lacks")
    base = w._font_combo.count()

    cfg = w.config()
    w.apply_config(replace(cfg, style=replace(cfg.style, font_family="Calibri")))
    assert w._font_combo.count() == base + 1 and w._font_off_list, "premise: the row is up"

    installed = w._font_combo.findData("Courier New")
    assert installed > 0, "premise: this machine has Courier New"
    steps = w._history.depth()[0]
    w._font_combo.setCurrentIndex(installed)            # the pick, through the widget

    assert w._style.font_family == "Courier New"
    assert w._font_combo.count() == base, "the absent family's row went with the decision"
    assert not w._font_off_list
    assert w._font_combo.findData("Calibri") == -1
    # Removing row 1 renumbers everything above it. The row the user chose must still be
    # the current one, and their one gesture must still be one undo step — an unblocked
    # removal would re-enter the handler and take a second.
    assert w._font_combo.currentData() == "Courier New"
    assert w._font_combo.currentText() == "Courier New"
    assert w._history.depth()[0] == steps + 1, "one gesture is still one undo step"

    # Undo is what the row was waiting for all along: the absent family comes back, and so
    # does the sentence that explains it.
    assert w.undo_config()
    assert w._style.font_family == "Calibri"
    assert w._font_off_list and "not on this computer" in w._font_combo.currentText()

    # Re-selecting THAT row is a choice too, and it keeps it.
    w._font_combo.setCurrentIndex(1)
    assert (w._style.font_family, w._font_off_list) == ("Calibri", True)
    assert w._font_combo.count() == base + 1

    # ...and the default row retires it exactly as any other pick does.
    w._font_combo.setCurrentIndex(0)
    assert (w._style.font_family, w._font_off_list) == ("", False)
    assert w._font_combo.count() == base


def test_the_font_tooltip_describes_the_default_without_naming_a_private_face(win):
    """The tooltip may only print a family name the dropdown below it actually offers.

    It used to interpolate ``QFontInfo(QFont()).family()`` once, at construction, so on
    macOS it read ".AppleSystemUIFont" — a leading-dot internal name, the same kind
    ``_font_families`` strips OUT of the list as system-internal, and one no reader can act
    on. It was also frozen at construction, so it described a font nothing was drawn in any
    more as soon as the application font changed.
    """
    w = win
    tip = w._font_combo.toolTip()
    assert pp._FONT_DEFAULT_LABEL in tip, "it still explains what the default row means"

    resolved = QtGui.QFontInfo(QtGui.QFont()).family()
    if w._font_rows.get(resolved.casefold()) is None:
        # The stock macOS case, under cocoa AND offscreen: the application font resolves to
        # a private family. Then it is described, not named.
        assert resolved not in tip, (
            f"the tooltip must not print {resolved!r} — the list below cannot offer it")
        assert not any(word.startswith(".") and len(word) > 3 for word in tip.split()), (
            "no leading-dot family name anywhere in the sentence")
    else:
        assert resolved in tip, "a family the list DOES offer is worth naming"

    # THE FRESHNESS HALF. Change the application font under the running window and ask for
    # the tooltip the way Qt does; the sentence has to be about the font in force now.
    app = QtWidgets.QApplication.instance()
    was = app.font()
    try:
        app.setFont(QtGui.QFont("Courier New", was.pointSize()))
        assert "Courier New" in w._font_row_tooltip(), "composed against the live app font"
        combo = w._font_combo
        QtWidgets.QApplication.sendEvent(combo, QtGui.QHelpEvent(
            QtCore.QEvent.ToolTip, QtCore.QPoint(5, 5),
            combo.mapToGlobal(QtCore.QPoint(5, 5))))
        assert "Courier New" in combo.toolTip(), (
            "the ToolTip event rebuilds the text before Qt reads it")
    finally:
        app.setFont(was)


def test_pointing_the_font_row_at_a_family_cannot_commit_it_whatever_the_caller_did(win):
    """``_select_font_family`` GUARANTEES the no-commit rule; it does not ask for it.

    The rule is that moving the Font row to show a style must never reach
    ``_on_style_font_family`` and write a family the user did not choose. It was enforced
    by an ``assert`` that the CALLER had blocked the combo's signals — which is no
    enforcement under ``python -O``, where the assert is compiled out and a call site that
    forgot would commit silently. This test drives the method with the signals wide open,
    so it fails on the old code in both modes: ``AssertionError`` without ``-O``, a
    committed family with it.
    """
    w = win
    assert not w._font_combo.signalsBlocked(), "premise: nobody has blocked anything"
    assert w._style.font_family == ""

    w._select_font_family("Courier New")

    assert w._font_combo.currentData() == "Courier New", "the row moved"
    assert w._style.font_family == "", "...and committed nothing"
    assert not w._font_combo.signalsBlocked(), "the combo is left able to speak again"

    # A caller that blocks for reasons of its own gets ITS state back, not a cleared one —
    # ``_apply_style_widgets`` writes ~40 controls and must stay in charge of the rest.
    w._font_combo.blockSignals(True)
    try:
        w._select_font_family("")
        assert w._font_combo.signalsBlocked(), "the previous blocked state is restored"
    finally:
        w._font_combo.blockSignals(False)
    assert (w._style.font_family, w._font_combo.currentIndex()) == ("", 0)
