"""S3 (repair) — every INSTANT column carries its clock and its unit, on the axis too.

JHT-05 gave the Plots window the rule ``core/clock.py`` states -- "One clock, or every
control that shows or takes a time says which clock it is in" -- and wired it through two
outputs: a CLOCK note in Details (driven by ``_ORIGIN_COLUMNS``) and a clause on the axis
label itself (driven by ``_TIME_COLUMNS``), the axis being the half an exported image
keeps. The two lists were spelled out separately, and they disagreed: ``t_end_s`` was in
the first and not the second. So a histogram of ``t_end_s`` printed the clock in Details
and drew

    t_end_s

as its axis -- no t₀, and no time unit either, because ``_TIME_COLUMNS`` is also what
makes the Time-unit combo reach a column (its axis scale, its printed parameters and its
Fit-range boxes) -- while ``t_start_s`` beside it read
``t_start (ms, from recording start (t₀ = 61.3034 s))``.

The lists are now derived, not restated: ``_ORIGIN_COLUMNS`` is
``analysis.scoring.RECORDING_CLOCK_COLUMNS`` -- the project's own declaration of which
columns are instants on the recording clock, the same set ``gui/panels`` badges the event
table with -- and ``_TIME_COLUMNS`` is durations + that. An instant column added to the
shared declaration arrives here already carrying its unit and its clock.

These tests read the RENDERED axis label off a real ``PlotsWindow``, on both draw paths
that have data axes (histogram and scatter).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import scoring
from nanoscope.core.session import Session
from nanoscope.gui import plots_panel
from nanoscope.gui.plots_panel import PlotsWindow

FS = 50_000.0
N = 400


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = np.sort(rng.uniform(0.0, 10.0, N))
    dwell = rng.exponential(2e-3, N) + 2e-5
    return pd.DataFrame({
        "t_start_s": t0,
        "t_end_s": t0 + dwell,
        "dwell_s": dwell,
        "rise_s": np.abs(rng.normal(8e-5, 2e-5, N)) + 4e-5,
        "depth_pA": np.abs(rng.normal(80.0, 12.0, N)),
        "accepted": np.ones(N, dtype=bool),
        "filter_limited": np.zeros(N, dtype=bool),
    })


def _window(qtbot, *, unit: str = "ms") -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run": _frame()}, {"run": {"min_dwell_s": 2e-5, "fs": FS}})
    w.set_time_unit(unit)
    return w


def _label(w: PlotsWindow, side: str) -> str:
    """The axis label as RENDERED (pyqtgraph wraps it in a styled span)."""
    return w._plot.getAxis(side).labelString()


# ================================================================================================
# the two lists cannot disagree any more
# ================================================================================================

def test_the_instant_columns_come_from_the_module_that_declares_them():
    assert plots_panel._ORIGIN_COLUMNS == tuple(scoring.RECORDING_CLOCK_COLUMNS)


def test_every_instant_column_is_a_time_column():
    """The hole that let ``t_end_s`` through: an instant that the unit/axis rule missed."""
    missing = [c for c in plots_panel._ORIGIN_COLUMNS if c not in plots_panel._TIME_COLUMNS]
    assert missing == [], missing
    # ...and the durations are still there, still origin-free.
    for col in ("dwell_s", "rise_s", "fall_s"):
        assert col in plots_panel._TIME_COLUMNS
        assert col not in plots_panel._ORIGIN_COLUMNS


# ================================================================================================
# the histogram axis
# ================================================================================================

@pytest.mark.parametrize("col", ["t_start_s", "t_end_s"])
def test_an_instant_axis_names_its_clock_and_its_unit(qapp, qtbot, col):
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    label = _label(w, "bottom")
    assert "recording start" in label, label      # the clock
    assert "(ms" in label, label                  # ...in the unit the combo is showing
    assert col.removesuffix("_s") in label, label  # ...for the column actually drawn


@pytest.mark.parametrize("col", ["dwell_s", "rise_s"])
def test_a_duration_axis_takes_the_unit_but_not_a_clock(qapp, qtbot, col):
    """CONTROL — a LENGTH of time is on no clock; annotating it would be noise."""
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    label = _label(w, "bottom")
    assert "(ms" in label, label
    assert "recording start" not in label and "t₀" not in label, label


def test_a_picoamp_axis_is_untouched(qapp, qtbot):
    """CONTROL — the time unit is a TIME unit; a depth axis must not take it."""
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    label = _label(w, "bottom")
    assert "ms" not in label and "recording start" not in label, label


# ================================================================================================
# the scatter axes -- both of them
# ================================================================================================

def test_both_scatter_axes_name_the_clock_for_instant_columns(qapp, qtbot):
    w = _window(qtbot)
    w.set_plot_type("Scatter")
    w.set_scatter_columns("t_start_s", "t_end_s")
    for side in ("bottom", "left"):
        label = _label(w, side)
        assert "recording start" in label, (side, label)
        assert "(ms" in label, (side, label)


def test_a_scatter_of_an_instant_against_a_duration_marks_only_the_instant(qapp, qtbot):
    w = _window(qtbot)
    w.set_plot_type("Scatter")
    w.set_scatter_columns("t_end_s", "dwell_s")
    x, y = _label(w, "bottom"), _label(w, "left")
    assert "recording start" in x, x
    assert "recording start" not in y, y


# ================================================================================================
# the two halves of the rule agree
# ================================================================================================

@pytest.mark.parametrize("col", ["t_start_s", "t_end_s"])
def test_details_and_the_axis_tell_the_same_story(qapp, qtbot, col):
    """Neither half alone: Details is what a reader sees, the axis is what an export keeps."""
    w = _window(qtbot)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    details = w.details_text()
    assert "CLOCK" in details or "t₀" in details, details
    assert "recording start" in _label(w, "bottom")
