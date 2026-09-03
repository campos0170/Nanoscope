"""C1/C2 and E1/E2: one log concept, an axis range that talks back, and fit statistics.

Four complaints, one file, because they are all about the same surface — the axis and
fit-DISPLAY half of the Plots window — and each is a case of the window doing something
without saying so:

* **C1.** Three independent per-page log-toggle pairs meant "is this axis logarithmic"
  was answered by switching on the plot type in four places, and the KDE page's Log-x
  default was written as a literal in two.
* **C2.** One all-or-nothing Auto button, and three typed instructions the Axis-range
  boxes declined in SILENCE — reversed limits, a non-positive bound on a log axis, and a
  half-filled pair — with the numbers still in the boxes and the axis behaving as if they
  were empty.
* **E1.** R² and the KS verdict were computed and printed with no way to keep them off a
  figure; there was no confidence BAND and no residuals at all.
* **E2.** The provenance is unusually rich for an ML run — model file, preset, decode
  preset, the whole preprocessing chain — and a saved PNG of a dwell histogram carried
  none of it, so three differently-trained models produced three indistinguishable
  figures.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import time
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import KdeConfig, PlotConfig, PlotsWindow


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 600, *, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key="doc0:0", label="apo −500 mV", prov=None, **kw) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(),
                      provenance=dict(prov or {"fs": 250_000.0, "min_dwell_s": 2e-5,
                                               "detector": "schmitt_hysteresis"}),
                      colour="#0072B2", doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold",
                      calibration_fingerprint=(1.0, 0.0), **kw)


#: An ML run's stamp, the shape ``ml_detect`` writes: the keys E2 exists to surface.
_ML_PROV = {
    "fs": 250_000.0, "min_dwell_s": 2e-5,
    "detector": "ml_keras",
    "model_path": "/Users/someone/models/zn_30h_student.keras",
    "ml_preset": "zn_v2",
    "ml_decode_preset": "zn_deep",
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.2,
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 3,
    "hp_enabled": True, "hp_cutoff_hz": 100.0,
    "normalize_enabled": True, "normalize_mode": "masked_median",
}


def _drain(pp: PlotsWindow, qtbot, *, timeout_s: float = 240.0) -> None:
    """Wait for the deferred notes to land, loudly if they never do — the same helper the
    sibling files use, for the same reason: a silent timeout is indistinguishable from a
    product regression."""
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
# C1 — a log choice is ONE concept
# ================================================================================================

def test_log_state_answers_for_whichever_page_is_on_screen():
    """Four call sites used to switch on the plot type to ask this. One does now."""
    cfg = PlotConfig()
    assert cfg.log_state == (False, False)
    assert replace(cfg, plot_type="Scatter").with_log_state(log_x=True).log_state == (True, False)
    kde = replace(cfg, plot_type="KDE (density)")
    assert kde.log_state == (kde.kde.log_x, kde.kde.log_y)
    assert replace(cfg, plot_type="PCA").log_state == (False, False), "scores are signed"


def test_with_log_state_writes_the_ACTIVE_page_and_leaves_the_others_alone():
    """Per-page memory is the storage; "is this axis logarithmic" is the question. A
    writer that spilled onto the other pages would silently log a plot nobody is looking
    at, and the round trip in test_plots_config would stop holding."""
    cfg = replace(PlotConfig(), plot_type="Scatter").with_log_state(log_x=True, log_y=True)
    assert cfg.scatter.log_x and cfg.scatter.log_y
    assert not cfg.histogram.log_x and not cfg.histogram.log_y
    # None leaves an axis alone rather than clearing it.
    assert cfg.with_log_state(log_x=False).log_state == (False, True)
    # PCA has no toggle to move; inventing one would be a control that cannot be honoured.
    pca = replace(PlotConfig(), plot_type="PCA")
    assert pca.with_log_state(log_x=True) == pca


def test_the_window_reads_its_log_state_through_the_config_not_a_fourth_switch(win):
    win.set_log_x(True)
    assert win._current_log_state() == (True, False) == win.config().log_state
    win.set_plot_type("Scatter")
    win.set_scatter_log(False, True)
    assert win._current_log_state() == (False, True) == win.config().log_state


def test_set_log_state_is_the_page_agnostic_setter(win):
    """A caller that means "log the x axis of what is on screen" no longer has to read
    the plot type first and choose between three setters."""
    win.set_log_state(log_x=True)
    assert win._hist_logx.isChecked()
    win.set_plot_type("Scatter")
    win.set_log_state(log_x=True, log_y=True)
    assert win._scatter_logx.isChecked() and win._scatter_logy.isChecked()
    assert not win._hist_logy.isChecked(), "the histogram page was not on screen"


def test_the_kde_log_x_default_is_stated_once(win):
    """It used to be a literal in the KdeConfig default AND a bare setChecked(True) in the
    page builder — two sources for one value, which is how a fresh window and a
    reset_state'd one come to disagree."""
    assert win._kde_logx.isChecked() == KdeConfig().log_x
    win.reset_state()
    assert win._kde_logx.isChecked() == KdeConfig().log_x
    assert win._kde_logx.toolTip(), "the default has to say it is a default"


# ================================================================================================
# C2 — per-axis Auto, and three refusals that used to be silent
# ================================================================================================

def test_auto_x_leaves_the_y_pin_alone(win):
    """The all-or-nothing button threw away a pin the user did not touch: pinning y to a
    fixed count ceiling while re-fitting x per pore was a five-gesture job."""
    win.set_axis_range(xmin=1e-4, xmax=2e-3, ymin=0.0, ymax=50.0)
    assert win._manual_x is not None and win._manual_y is not None

    win._reset_axis_range(axis="x")
    assert win._manual_x is None, "X was unpinned"
    assert win._manual_y == pytest.approx((0.0, 50.0)), "Y was not touched"
    assert win._range_edits["xmin"].text() == "" and win._range_edits["ymax"].text() != ""

    win._reset_axis_range(axis="y")
    assert win._manual_y is None
    assert all(e.text() == "" for e in win._range_edits.values())


def test_auto_still_means_both(win):
    win.set_axis_range(xmin=1e-4, xmax=2e-3, ymin=0.0, ymax=50.0)
    win._reset_axis_range(axis="both")
    assert win._manual_x is None and win._manual_y is None


def test_a_half_filled_pair_says_so_instead_of_silently_unpinning_the_axis(win):
    win._range_edits["xmin"].setText("0.1")
    win._range_edits["xmax"].setText("")
    win._apply_axis_range()

    assert win._manual_x is None, "the behaviour is unchanged — both edges pin an axis"
    notes = win._axis_range_input_notes()
    assert len(notes) == 1 and "X max" in notes[0] and "blank" in notes[0]
    assert "IGNORED" in notes[0] and "BOTH of its edges" in notes[0]
    # `isVisible()` is False for every widget of a window that was never shown, so the
    # assertion is on `isHidden()` — which is true only of a widget hidden on purpose.
    assert not win._range_note.isHidden() and notes[0] in win._range_note.text()


def test_reversed_limits_say_so(win):
    win._range_edits["ymin"].setText("50")
    win._range_edits["ymax"].setText("1")
    win._apply_axis_range()

    assert win._manual_y is None
    notes = win._axis_range_input_notes()
    assert len(notes) == 1 and "Y range [50, 1]" in notes[0] and "min ≥ max" in notes[0]


def test_a_non_positive_bound_on_a_LOG_axis_says_which_of_the_two_to_change(win):
    """0 is exactly what a person types for "start at the origin", and log10 of it is not
    a number. The refusal is right; the silence left no way to guess the fix."""
    win.set_log_x(True)
    win._range_edits["xmin"].setText("0")
    win._range_edits["xmax"].setText("2")
    win._apply_axis_range()

    notes = win._axis_range_input_notes()
    assert len(notes) == 1 and "not positive" in notes[0] and "logarithmic" in notes[0]
    assert "untick Log x" in notes[0]
    # ...and it is a statement about the LOG axis, not about the number: untick it and
    # the same pair is accepted.
    win.set_log_x(False)
    win._apply_axis_range()
    assert win._axis_range_input_notes() == []
    assert win._range_note.isHidden()
    assert win._manual_x is not None, "the same pair pins fine on a linear axis"


def test_a_usable_pair_says_nothing(win):
    """The line is amber and must not be on screen at rest, or it stops being read."""
    win.set_axis_range(xmin=1e-4, xmax=2e-3)
    assert win._axis_range_input_notes() == []
    assert win._range_note.isHidden()


def test_the_note_names_the_unit_the_box_is_read_in(win):
    """"min ≥ max" over a pair of boxes labelled in µs is only actionable if you know
    which pair and which unit."""
    win.set_time_unit("µs")
    win._range_edits["xmin"].setText("2000")
    win._range_edits["xmax"].setText("100")
    win._apply_axis_range()
    assert "(µs)" in win._axis_range_input_notes()[0]


def test_restoring_a_config_restores_its_explanation(win):
    """An undo that puts a reversed pair back must put its amber line back with it."""
    win._range_edits["xmin"].setText("5")
    win._range_edits["xmax"].setText("1")
    win._apply_axis_range()
    broken = win.config()

    win._reset_axis_range(axis="both")
    assert win._range_note.isHidden()

    win.apply_config(broken)
    assert not win._range_note.isHidden()
    assert "min ≥ max" in win._range_note.text()


# ================================================================================================
# E1 — the fit statistics, and the switches that cannot hide a refusal
# ================================================================================================

def test_the_four_switches_default_to_todays_figure():
    """Two statistics were already printed unconditionally and two pictures are new, so
    the defaults are asymmetric on purpose: an untouched window renders as it did."""
    s = PlotConfig().style
    assert s.show_r2 is True and s.show_gof is True
    assert s.show_ci_band is False and s.show_residuals is False
    assert s.show_provenance is False


def test_the_switches_round_trip_through_the_widgets(win):
    """They are style, so they are config, so undo/redo and templates carry them."""
    cfg = replace(win.config(), style=replace(
        win.config().style, show_r2=False, show_gof=False,
        show_ci_band=True, show_residuals=True, show_provenance=True))
    win._set_config(cfg, draw=False)
    assert win.config() == cfg
    assert win._r2_check.isChecked() is False and win._ci_band_check.isChecked() is True


def _narrow_peak_events(n: int = 400, *, seed: int = 11) -> pd.DataFrame:
    """Dwell times that are a NARROW PEAK, not an exponential: the shape of the owner's
    -600 mV apo-ferritin recording (quartiles 16.61 / 18.14 / 19.48 ms there), on which
    the app's own KS test rejects the exponential at D = 0.3966 against 0.0981."""
    rng = np.random.default_rng(seed)
    dwell = np.clip(rng.normal(0.018, 0.0015, n), 0.005, None)
    return pd.DataFrame({
        "dwell_s": dwell,
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _drawn_title(win) -> str:
    import re
    return re.sub(r"<[^>]+>", "", str(win._plot.titleLabel.text)).strip()


def _drawn_legend(win) -> list[str]:
    legend = getattr(win._plot, "legend", None)
    return [] if legend is None else [label.text for _s, label in legend.items]


@pytest.mark.parametrize("show_gof", [True, False])
@pytest.mark.parametrize("values_box", [False, True])
def test_turning_the_KS_verdict_off_cannot_hide_a_REJECTION(qapp, qtbot, show_gof, values_box):
    """Off means "stop decorating a figure whose fit is fine". A curve the test threw out
    is the difference between a measurement and a decoration, and says so either way —
    ON THE FIGURE, in the surfaces Save image… exports.

    THIS TEST USED TO ASSERT ON ``_full_fit_summary``, THE STRING BUILDER, and never on
    what was drawn — which is how it passed while the app's own shipped dwell template,
    at its defaults (``show_gof`` True, ``fit_values_on_plot`` False), exported the
    owner's -600 mV recording as title "τ = 20.41 ms (n = 275)" and legend "fit: τ =
    20.41 ms (n = 275)" for a curve rejected at p = 1.6e-39, with no verdict anywhere
    on the figure (S7-JRN2): the builder's output went only into the values box, which
    a third switch keeps off by default. So this drives the real window over a narrow
    peak, and reads the drawn TITLE, the drawn LEGEND entry and the values box, at all
    four settings of the two switches."""
    win = PlotsWindow(Session())
    qtbot.addWidget(win)
    win.set_data([PlotSource(key="doc0:0", label="apo −600 mV", df=_narrow_peak_events(),
                             provenance={"fs": 250_000.0, "min_dwell_s": 2e-5,
                                         "detector": "schmitt_hysteresis"},
                             colour="#0072B2", doc_id="doc0", channel=0,
                             module_id="detect.threshold",
                             calibration_fingerprint=(1.0, 0.0))])
    win.set_plot_type("Histogram")
    win.set_histogram_column("dwell_s")
    win.set_time_unit("ms")
    win._style = replace(win._style, show_gof=show_gof, fit_values_on_plot=values_box)
    win.set_fit("exponential")
    win._redraw()

    gof = win._last_fit.meta.get("gof")
    assert gof and gof.get("describes_data") is False, (
        f"premise: the exponential is REJECTED on a narrow peak -- {gof}")

    title = _drawn_title(win)
    assert "τ =" in title and "KS REJECTED" in title, title
    fit_rows = [t for t in _drawn_legend(win) if t.startswith("fit:")]
    assert fit_rows and fit_rows[0].endswith("KS REJECTED (1%)"), (
        fit_rows, "the legend entry is what a reader of the saved image sees")
    if values_box:
        assert win._fit_values_item is not None
        assert "REJECTED at the 1% level" in win._fit_values_item.text
    else:
        assert win._fit_values_item is None, "the box is off, and the verdict did not need it"
    assert "DOES NOT DESCRIBE THESE DATA" in win.details_text()


def test_a_fit_the_test_accepts_carries_no_verdict_in_the_title_or_the_legend(win):
    """The other direction: the tag is a REFUSAL, not a decoration, so a curve that passes
    puts nothing extra on the title or the legend, and the KS number reaches the figure
    only through the values box when that box is asked for -- which is all ``show_gof``
    governs."""
    win._style = replace(win._style, show_gof=True, fit_values_on_plot=False)
    win.set_fit("exponential")
    win._redraw()
    gof = win._last_fit.meta.get("gof")
    assert gof and gof.get("describes_data") is True, f"premise: a fit that passes -- {gof}"
    assert "REJECTED" not in _drawn_title(win)
    assert not any("REJECTED" in t for t in _drawn_legend(win))
    assert win._fit_values_item is None

    win._style = replace(win._style, fit_values_on_plot=True)
    win._redraw()
    assert "consistent with these data" in win._fit_values_item.text
    win._style = replace(win._style, show_gof=False)
    win._redraw()
    assert "KS" not in win._fit_values_item.text, "off means: stop decorating a fine fit"


def test_the_string_builder_still_carries_the_rejection_whatever_show_gof_says(win):
    """The old assertion, kept as the builder's own contract (the values box takes it)."""
    fit = type("F", (), {"params": {"tau_1": 3e-4}, "n": 500, "bic": 1.0,
                         "unit_of": lambda self, k: "s"})()
    win._style = replace(win._style, show_gof=False)

    passed = {"ks": 0.02, "describes_data": True}
    rejected = {"ks": 0.31, "describes_data": False}
    assert "KS" not in win._full_fit_summary(fit, "dwell_s", passed)
    assert "REJECTED at the 1% level" in win._full_fit_summary(fit, "dwell_s", rejected)

    win._style = replace(win._style, show_gof=True)
    assert "KS = 0.020" in win._full_fit_summary(fit, "dwell_s", passed)


def test_the_R2_switch_keeps_every_sentence_that_says_the_R2_is_not_evidence(win):
    """An R² through two points is exact by construction, and an R² over a flat response
    is undefined. Both are refusals, not statistics, so neither is optional."""
    win.set_plot_type("Scatter")
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_scatter_fit("linear")
    win._style = replace(win._style, show_r2=False)
    win._redraw()
    with_off = win.details_text()

    win._style = replace(win._style, show_r2=True)
    win._redraw()
    with_on = win.details_text()

    assert "R²" in with_on
    assert "R² =" not in with_off, "the plain number is what the switch turns off"
    assert "y = " in with_off, "the equation still stands on the figure"


def test_the_residual_strip_appears_only_when_asked_and_carries_its_chi_square(win):
    win.set_fit("exponential")
    win._redraw()
    assert win._resid_widget.isHidden(), "off by default — a second axis is clutter"

    win._style = replace(win._style, show_residuals=True)
    win._redraw()
    assert not win._resid_widget.isHidden()
    assert win._resid.items, "the strip has residuals in it"
    details = win.details_text()
    assert "χ²" in details and "APPROXIMATE" in details, details[-400:]
    assert "no p-value" in details

    win._style = replace(win._style, show_residuals=False)
    win._redraw()
    assert win._resid_widget.isHidden()


def test_the_strips_number_reaches_the_exported_figure_even_though_the_strip_does_not(win):
    """Save image… exports the main axes only, so the reportable half — the reduced χ² —
    is written into the on-plot values box, which the PNG does keep."""
    fit = type("F", (), {"params": {"tau_1": 3e-4}, "n": 500, "bic": 1.0,
                         "unit_of": lambda self, k: "s"})()
    box = win._full_fit_summary(fit, "dwell_s", None, "χ²/dof = 1.32 (approx. dof)")
    assert "χ²/dof = 1.32 (approx. dof)" in box
    assert "approx" in box, "a bare χ²/dof reads as the textbook quantity"


def test_the_confidence_band_is_drawn_only_when_asked(win, qtbot):
    """The band comes from the deferred whole-fit job — the bootstrap runs on the worker
    and the FitResult it caches is the one carrying the replicates — so it appears on the
    redraw AFTER the notes settle, not on the click."""
    win.set_fit("exponential")
    _drain(win, qtbot)
    win._redraw()
    assert win._band_items == [], "off by default"

    win._style = replace(win._style, show_ci_band=True)
    _drain(win, qtbot)
    win._redraw()
    assert len(win._band_items) == 3, "upper, lower and the fill between them"
    assert any("POINTWISE" in line for line in win.details_text().splitlines()), (
        "a band the reader could mistake for a simultaneous one has to say which it is")

    win._style = replace(win._style, show_ci_band=False)
    win._redraw()
    assert win._band_items == [], "and it does not survive into the next picture"


def test_a_fit_with_no_replicates_draws_nothing_rather_than_a_fabricated_band(win):
    """Four ways a fit legitimately has none (a reopened project, point_fits_only, the
    exp+gamma family, too few usable replicates). Every one of them must be a band-less
    figure, never the curve redrawn at the corners of the parameter box."""
    from nanoscope.analysis import stats

    rng = np.random.default_rng(1)
    t = 2e-5 + rng.exponential(3e-4, 400)
    with stats.point_fits_only():
        bare = stats.fit_dwell_exponential(t, t_min=2e-5)
    grid = np.geomspace(t.min(), t.max(), 50)
    assert win._draw_ci_band(bare, grid, grid, 1.0, "#d55e00") == ""
    assert win._band_items == []


def test_the_band_is_not_a_second_legend_row(win):
    """One object, one entry — the note-Q2 complaint the legend checklist exists to fix."""
    win.set_fit("exponential")
    win._style = replace(win._style, show_ci_band=True)
    win._redraw()
    assert not any("band" in str(l).lower() for l in win.legend_labels())


# ================================================================================================
# E2 — the figure says what made it
# ================================================================================================

def test_the_caption_names_the_detector_the_model_and_the_chain(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source(prov=_ML_PROV)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")

    line = w._provenance_line(w.sources()[0])
    assert "ML (Keras)" in line
    assert "model zn_30h_student.keras" in line, "the FILE is what tells HG/Zn/Cu apart"
    assert "/Users/" not in line, "a home directory does not belong in a published figure"
    assert "preset zn_v2" in line and "decode zn_deep" in line
    for stage in ("detrend block_median 0.2 s", "notch 60 Hz ×3",
                  "high-pass 100 Hz", "normalize masked_median"):
        assert stage in line, (stage, line)


def test_the_chain_lists_the_stages_that_RAN(win):
    """Every *_enabled key is a decision the module stamped, so the chain describes this
    recording rather than the panel's defaults."""
    assert win._preprocessing_chain({}) == ""
    assert win._preprocessing_chain(
        {"detrend_enabled": False, "notch_enabled": True,
         "notch_freqs": [60.0, 180.0]}) == "notch 60, 180 Hz"
    assert "no preprocessing" in win._provenance_line(win.sources()[0])


def test_an_unknown_detector_is_printed_verbatim_rather_than_dropped(win):
    """A detector this table does not know is exactly the case where the reader most
    needs to be told which one ran."""
    src = _source(prov={"detector": "some_future_detector"})
    assert "some_future_detector" in win._provenance_line(src)
    assert "detector not recorded" in win._provenance_line(_source(prov={"fs": 1.0}))


def test_the_caption_is_off_by_default_and_lands_on_the_figure_when_asked(win):
    win._redraw()
    assert win._prov_item is None

    win._style = replace(win._style, show_provenance=True)
    win._redraw()
    assert win._prov_item is not None
    # Parented to the ViewBox, which is what puts it in the exported image and keeps it
    # still under pan and zoom — the same rule the fit-values box follows.
    assert win._prov_item.parentItem() is win._plot.vb


def test_the_caption_declines_to_draw_over_a_message(win):
    """"No data yet" with a preprocessing chain under it describes nothing."""
    win._style = replace(win._style, show_provenance=True)
    win._sources = {}
    win._redraw()
    assert win._prov_item is None


def test_an_overlay_attributes_each_line_to_its_own_source(qapp, qtbot):
    """provenance is per source — min_dwell_s and a model file are properties of the run
    that made THOSE rows — so an unattributed line on a two-source plot names nothing."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source(key="doc0:0", label="zn"),
                _source(key="doc1:0", label="cu", prov=_ML_PROV)])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_sources(["zn", "cu"])
    w._style = replace(w._style, show_provenance=True)
    w._redraw()
    assert w._prov_item is not None
    text = w._prov_item.text
    assert "zn:" in text and "cu:" in text
