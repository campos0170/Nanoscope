"""Regression tests for EXPL-1: the scatter best-fit equation follows the axes' units.

``_overlay_scatter_fit`` used to print slope/intercept (and the power law's
coefficient) in RAW storage units — seconds for the time columns — under an axis whose
ticks show ms/µs, with no unit stated anywhere: "y = 2000·x + 5" over "dwell (ms)" for
data lying exactly on depth = 2 pA/ms · dwell + 5 pA. A reader taking x off the axis
computed depths 1000x too large. The fix routes the caption through the same
:meth:`PlotsWindow._display_unit` rule the exponential's tau and the mixture's mu
already use: slope and intercept rescale as slope·fy/fx and intercept·fy, the power
law's coefficient as fy/fx^b, and the units are SAID in the caption itself.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow


# ================================================================================================
# builders
# ================================================================================================

def _frame(**cols) -> pd.DataFrame:
    """An events-shaped frame: the given columns plus the two boolean flags."""
    n = len(next(iter(cols.values())))
    return pd.DataFrame({
        **cols,
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _dwell_vs_depth() -> pd.DataFrame:
    """depth_pA = 2 pA/ms · dwell + 5 pA, stored (as always) in seconds and pA."""
    dwell = np.linspace(2e-4, 3e-3, 60)          # 0.2 – 3 ms
    return _frame(dwell_s=dwell, depth_pA=2000.0 * dwell + 5.0)


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


def _scatter(window: PlotsWindow, df: pd.DataFrame, x: str, y: str, kind: str) -> None:
    window.set_data({"Detected events": df})
    window.set_plot_type("Scatter")
    window.set_scatter_columns(x, y)
    window.set_scatter_fit(kind)


def _texts(window: PlotsWindow) -> tuple[str, str]:
    return window.plot_item().titleLabel.text, window.details_text()


# ================================================================================================
# linear fit: the printed slope/intercept are true for numbers read off the ticks
# ================================================================================================

def test_linear_fit_caption_follows_ms_axis(window):
    _scatter(window, _dwell_vs_depth(), "dwell_s", "depth_pA", "linear")
    # The axis really is showing milliseconds (default time unit, ticks x1000)...
    axis = window.plot_item().getAxis("bottom")
    assert "dwell (ms)" in axis.labelText
    assert abs(axis.scale - 1e3) < 1e-9
    # ...and the equation matches those ticks: 2 pA per ms, units stated.
    for text in _texts(window):
        assert "y = 2·x + 5" in text
        assert "(x in ms, y in pA)" in text
        assert "2000·x" not in text                       # the raw per-second slope


def test_linear_fit_caption_follows_us_axis(window):
    _scatter(window, _dwell_vs_depth(), "dwell_s", "depth_pA", "linear")
    window.set_time_unit("µs")
    for text in _texts(window):
        assert "y = 0.002·x + 5" in text                  # 2000 pA/s == 0.002 pA/µs
        assert "(x in µs, y in pA)" in text


def test_linear_fit_caption_in_seconds_still_states_units(window):
    # With the axis in raw seconds the NUMBER was always right; the unit must
    # still be said, or the caption is one axis-relabel away from lying again.
    _scatter(window, _dwell_vs_depth(), "dwell_s", "depth_pA", "linear")
    window.set_time_unit("s")
    for text in _texts(window):
        assert "y = 2000·x + 5" in text
        assert "(x in s, y in pA)" in text


def test_linear_fit_converts_a_time_column_on_the_y_axis(window):
    # dwell_s = 0.002 s/pA · depth + 0.001 s  ==  2 ms/pA · depth + 1 ms.
    depth = np.linspace(5.0, 50.0, 40)
    df = _frame(depth_pA=depth, dwell_s=2e-3 * depth + 1e-3)
    _scatter(window, df, "depth_pA", "dwell_s", "linear")
    for text in _texts(window):
        assert "y = 2·x + 1" in text
        assert "(x in pA, y in ms)" in text


def test_linear_fit_time_on_both_axes(window):
    # rise = 0.5 · dwell: the factors cancel in the slope, both units are named.
    dwell = np.linspace(2e-4, 3e-3, 40)
    df = _frame(dwell_s=dwell, rise_s=0.5 * dwell)
    _scatter(window, df, "dwell_s", "rise_s", "linear")
    for text in _texts(window):
        assert "y = 0.5·x" in text
        assert "(x in ms, y in ms)" in text


# ================================================================================================
# power law: the coefficient's x-unit dependence carries the exponent (a·fy/fx^b)
# ================================================================================================

def test_power_law_caption_follows_ms_axis(window):
    # depth = 3e6 · dwell²  in (s, pA)  ==  3 · x²  in (ms, pA).
    dwell = np.linspace(2e-4, 3e-3, 60)
    df = _frame(dwell_s=dwell, depth_pA=3e6 * dwell ** 2)
    _scatter(window, df, "dwell_s", "depth_pA", "power law")
    for text in _texts(window):
        assert "y = 3·x^2" in text
        assert "(x in ms, y in pA)" in text
        assert "3e+06" not in text                        # the raw per-second coefficient


def test_power_law_exponent_is_untouched_by_the_unit(window):
    # b is dimensionless: switching the display unit rescales the coefficient only.
    dwell = np.linspace(2e-4, 3e-3, 60)
    df = _frame(dwell_s=dwell, depth_pA=3e6 * dwell ** 2)
    _scatter(window, df, "dwell_s", "depth_pA", "power law")
    window.set_time_unit("s")
    title, details = _texts(window)
    assert "x^2" in title
    assert "y = 3e+06·x^2" in title
    assert "(x in s, y in pA)" in title
    assert "y = 3e+06·x^2" in details


# ================================================================================================
# columns with no known unit keep the plain caption (factor 1, no unit note)
# ================================================================================================

def test_unitless_columns_keep_the_plain_caption(window):
    x = np.linspace(1.0, 10.0, 40)
    df = _frame(x=x, lin=2.0 * x + 1.0)
    _scatter(window, df, "x", "lin", "linear")
    for text in _texts(window):
        assert "y = 2·x + 1" in text
        assert "(x in" not in text
