"""Sweep-5 numeric-edge regressions NE-05 / NE-06: LINEAR tick labels.

pyqtgraph's default linear ``tickStrings`` follows the tick MAGNITUDE, not the span:
any ``|value| >= 10000`` falls to ``"%g"`` with 6 significant digits. Two shipped
symptoms, one root:

* NE-05 — a trace time axis a day into a recording (t0 = 86,400 s) zoomed to a
  100 µs window labelled EVERY major tick ``86400``: six identical labels on the one
  surface whose job is saying WHERE in the recording you are.
* NE-06 — a 10⁶ pA autorange printed ``800000`` beside ``1e+06`` on one axis: two
  notations for the same KIND of number, the defect class ``gui/axes.py`` already
  refuses in log mode.

The invariant now lives in :class:`nanoscope.gui.axes.ReadableLogAxis.tickStrings`
(linear mode) and its linear-surface alias :class:`ReadableLinearAxis`: label
precision derives from the tick SPACING, and the fixed-vs-``m·10ⁿ`` notation is
chosen once per axis. These tests pin the dossier's exact scenarios plus the
properties the fix rests on.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.gui.axes import ReadableLinearAxis, ReadableLogAxis

# A label is "exponent-notation" iff it carries the ·10ⁿ (or a %g-style e+NN) marker.
_EXP = re.compile(r"·10|e[+-]\d")


def _notations(labels: list[str]) -> set[str]:
    """{'plain'} / {'exp'} / both — '0' is notation-neutral by convention."""
    kinds = set()
    for s in labels:
        if not s or s == "0":
            continue
        kinds.add("exp" if _EXP.search(s) else "plain")
    return kinds


@pytest.fixture
def axis(qapp):
    return ReadableLinearAxis(orientation="bottom")


# ================================================================================================
# NE-05: the dossier's day-deep zoom — labels must DISTINGUISH the ticks
# ================================================================================================

def test_ne05_day_deep_100us_window_labels_are_distinct(axis):
    """t0 = 86,400 s, 100 µs window, 2e-5 s spacing: the shipped axis said '86400' x6."""
    vals = [86400.0 + i * 2e-5 for i in range(6)]
    labels = axis.tickStrings(vals, 1.0, 2e-5)
    assert len(set(labels)) == len(labels), labels
    # and they are readable positions, not scientific residue
    assert labels[0] == "86400.00000"
    assert labels[1] == "86400.00002"
    assert _notations(labels) == {"plain"}


def test_ne05_no_exotic_t0_needed_a_long_recording_reaches_it_too(axis):
    """The verifier's alternate path: t0=0, 10 ms window at t=12,345 s of a plain
    long recording collapsed identically. Span-derived precision fixes both."""
    vals = [12345.0 + i * 2e-3 for i in range(6)]
    labels = axis.tickStrings(vals, 1.0, 2e-3)
    assert len(set(labels)) == len(labels), labels
    assert all(lab.startswith("12345.0") for lab in labels)


def test_ne05_precision_follows_spacing_not_magnitude(axis):
    """The same magnitude under a WIDE span keeps integer labels — no digit spam."""
    vals = [0.0, 20000.0, 40000.0, 60000.0, 86400.0]
    labels = axis.tickStrings(vals, 1.0, 20000.0)
    assert labels == ["0", "20000", "40000", "60000", "86400"]


# ================================================================================================
# NE-06: one notation per axis
# ================================================================================================

def test_ne06_the_1e6_pa_axis_never_mixes_notations(axis):
    """The dossier's y autorange: ['0','200000',...,'1e+06'] mixed %g into plain."""
    vals = [0.0, 2e5, 4e5, 6e5, 8e5, 1e6]
    labels = axis.tickStrings(vals, 1.0, 2e5)
    assert labels == ["0", "200000", "400000", "600000", "800000", "1000000"]
    assert _notations(labels) == {"plain"}


def test_ne06_negative_ranges_hold_the_same_invariant(axis):
    labels = axis.tickStrings([-1e6, -5e5, 0.0, 5e5, 1e6], 1.0, 5e5)
    assert len(_notations(labels)) == 1, labels
    assert len(set(labels)) == 5


def test_astronomical_values_switch_the_WHOLE_axis_to_exponent_form(axis):
    """Beyond MAX_PLAIN_DIGITS the axis flips to m·10ⁿ — for every tick at once,
    because a per-tick flip is exactly the mixed notation being pinned out."""
    vals = [0.0, 2e15, 4e15, 6e15, 1e16]
    labels = axis.tickStrings(vals, 1.0, 2e15)
    assert _notations(labels) == {"exp"}, labels
    assert len(set(labels)) == len(labels)


def test_fractional_spacing_keeps_ticks_distinguishable(axis):
    """pyqtgraph emits 2.5×10ⁿ spacings; one derived decimal too few would label
    0.25 as 0.2 while claiming a 0.3 neighbour."""
    labels = axis.tickStrings([0.0, 0.25, 0.5, 0.75], 1.0, 0.25)
    assert labels == ["0.00", "0.25", "0.50", "0.75"]


def test_no_minus_zero_label(axis):
    labels = axis.tickStrings([-1e-20, 0.0, 0.25], 1.0, 0.25)
    assert labels[0] == "0.00" and labels[1] == "0.00"


# ================================================================================================
# the fix must not disturb log mode, the PSD/plots windows' existing surface
# ================================================================================================

def test_log_mode_still_dispatches_to_logTickStrings(qapp):
    ax = ReadableLogAxis(orientation="bottom")
    ax.setLogMode(x=True)   # bottom axis follows x
    labels = ax.tickStrings([0.0, 1.0, 2.0], 1.0, 1.0)
    assert labels == ["10⁰", "10¹", "10²"]


def test_the_notes_patch_wiring_a_plotitem_with_axisItems_shows_real_positions(qapp):
    """The exact wiring the trace panes need (``axisItems={"bottom": ...}``), driven
    at the dossier's view range through a real PlotItem: the drawn axis's own
    tickValues/tickStrings pipeline must produce distinct labels."""
    plot = pg.PlotItem(axisItems={"bottom": ReadableLinearAxis(orientation="bottom"),
                                  "left": ReadableLinearAxis(orientation="left")})
    ax = plot.getAxis("bottom")
    lo, hi = 86400.0, 86400.0001
    tick_levels = ax.tickValues(lo, hi, 800)
    spacing, vals = tick_levels[0]
    labels = ax.tickStrings(vals, 1.0, spacing)
    assert len(vals) >= 3
    assert len(set(labels)) == len(labels), labels

    yax = plot.getAxis("left")
    tick_levels = yax.tickValues(-44387.0, 1.0444e6, 600)
    spacing, vals = tick_levels[0]
    labels = yax.tickStrings(vals, 1.0, spacing)
    assert len(_notations(labels)) <= 1, labels


def test_degenerate_spacing_falls_back_without_raising(axis):
    # pyqtgraph always passes a positive spacing; a direct degenerate call must not
    # crash the paint path.
    assert axis.tickStrings([1.0, 2.0], 1.0, 0.0) == ["1", "2"]
    assert axis.tickStrings([float("nan")], 1.0, 1.0) == [""]
