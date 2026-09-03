"""Beta EXPL-2: a fit refusal must name the column the user actually fitted.

``stats.fit_depth_gaussians`` used to call ``_clean_1d(frac_depth, "frac_depth", ...)``
with the name hardcoded, while the Plots window routes ANY histogram column
(``depth_pA``, ``amplitude``, ``snr``, ...) through it. Refusing a 3-event
``depth_pA`` fit therefore told the user

    Fit unavailable: frac_depth has only 3 events; refusing to fit fewer than 20.

-- a column they never selected and that may not even exist in their frame. The
``quantity`` parameter was added (report High-4) precisely so the fit knows what it
was handed; these tests pin that the refusals quote the same name the mu/sigma
captions already carry, at both the stats level (where the message is worded) and
through the real ``PlotsWindow`` widgets (where the user reads it).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import stats
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow


# ================================================================================================
# builders
# ================================================================================================

def _tiny_events(col: str, values) -> pd.DataFrame:
    """A 3-event frame carrying only the fitted column plus the schema flags.

    Three rows is below ``stats._MIN_EVENTS`` so the gaussian-mixture fit MUST
    refuse -- the point under test is which column the refusal blames.
    """
    n = len(values)
    return pd.DataFrame({
        col: list(values),
        "dwell_s": np.linspace(1e-4, 3e-4, n),
        "accepted": [True] * n,
        "filter_limited": [False] * n,
    })


def _refusal_line(window: PlotsWindow) -> str:
    lines = [l for l in window.details_text().splitlines() if "Fit unavailable" in l]
    assert lines, "the too-few-events fit must refuse, and say so in Details"
    return lines[0]


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


# ================================================================================================
# GUI: the Details pane names the column on the axis, not frac_depth
# ================================================================================================

@pytest.mark.parametrize("col", ["depth_pA", "amplitude", "snr"])
def test_gui_refusal_names_the_selected_column(window, col):
    """Histogram a non-frac_depth column, fit gaussian-mixture, 3 rows: the
    refusal quotes THAT column. 'frac_depth' -- never selected, not in the
    frame -- must not appear (beta EXPL-2)."""
    window.set_data({"Detected events": _tiny_events(col, [30.0, 31.0, 32.0])})
    window.set_plot_type("Histogram")
    window.set_histogram_column(col)
    window.set_fit("gaussian-mixture")
    refusal = _refusal_line(window)
    assert f"{col} has only 3 events" in refusal
    assert "frac_depth" not in refusal


def test_gui_frac_depth_refusal_still_says_frac_depth(window):
    """The one column whose refusal SHOULD say frac_depth still does: the fix
    threads the real name through, it does not rename the default."""
    window.set_data({"Detected events": _tiny_events("frac_depth", [0.2, 0.25, 0.3])})
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    window.set_fit("gaussian-mixture")
    assert "frac_depth has only 3 events" in _refusal_line(window)


# ================================================================================================
# stats: every refusal on the _clean_1d path quotes `quantity`
# ================================================================================================

def test_too_few_events_refusal_quotes_quantity():
    with pytest.raises(ValueError, match=r"depth_pA has only 3 events"):
        stats.fit_depth_gaussians(np.array([30.0, 31.0, 32.0]), quantity="depth_pA")


def test_default_quantity_still_reads_frac_depth():
    """No `quantity` passed (library callers fitting the frac_depth column):
    the wording is unchanged."""
    with pytest.raises(ValueError, match=r"frac_depth has only 3 events"):
        stats.fit_depth_gaussians(np.array([0.2, 0.25, 0.3]))


def test_dimensionless_default_quantity_maps_back_to_the_column_name():
    """The GUI passes quantity='dI/I0' for the frac_depth column (that is what
    the column MEASURES); the refusal quotes the column's name, which is what
    the axis and the picker show."""
    with pytest.raises(ValueError, match=r"frac_depth has only 3 events"):
        stats.fit_depth_gaussians(np.array([0.2, 0.25, 0.3]),
                                  quantity=stats.DEFAULT_DEPTH_QUANTITY)


def test_non_finite_refusal_quotes_quantity():
    x = np.r_[np.abs(np.random.default_rng(0).normal(30.0, 2.0, 30)), np.nan]
    with pytest.raises(ValueError, match=r"max_dev_pA contains 1 non-finite"):
        stats.fit_depth_gaussians(x, quantity="max_dev_pA")


def test_negative_depth_refusal_quotes_quantity():
    """The sign-convention refusal sits beside _clean_1d on the same path and
    had the same hardcoded name."""
    x = np.r_[np.abs(np.random.default_rng(0).normal(30.0, 2.0, 30)), -1.0]
    with pytest.raises(ValueError, match=r"negative depth_pA value"):
        stats.fit_depth_gaussians(x, quantity="depth_pA")


def test_identical_values_refusal_quotes_quantity():
    with pytest.raises(ValueError, match=r"all amplitude values are identical"):
        stats.fit_depth_gaussians(np.full(50, 30.0), quantity="amplitude")
