"""S7-FIX3 — "unit unknown" is not "dimensionless" in the density note.

``_column_quantity`` answers ``""`` both for a column the schema DECLARES dimensionless
(``frac_depth``, ``snr``, ``n_levels``, spelled ``"-"`` there) and for a column nobody
has declared at all, and ``_density_unit_notes`` collapsed both into "— a dimensionless
density". Measured on the manual-measurement table (Analysis ▸ Measure ▸ Add to list),
whose own ``units`` column read ``pA``: 14 of 14 of its ``signal``/``signal*s``/
``sample`` columns were told "bin width in the units of amplitude — a dimensionless
density" over heights that are per picoamp, while the same quantity off the events
table (``depth_pA``) read "a density in 1/pA".

The narrow repair the skeptic asked for: keep the dimensionless clause ONLY for a
declared ``"-"``; an undeclared unit says it is undeclared. NOT resolved from
``measure.MEASUREMENT_SCHEMA``, whose ``"signal"`` is a placeholder and which declares
``var`` as ``signal`` where the variance of a picoamp signal is pA².
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


@pytest.fixture
def window(qapp, qtbot):
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "amplitude": rng.normal(120.0, 25.0, n),         # a manual-measurement column: no unit declared
        "var": rng.normal(600.0, 50.0, n),               # declared "signal" by the measurement schema
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),    # declared pA
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)), # declared "-"
        "dwell_s": np.abs(rng.normal(0.02, 0.005, n)),
    })
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"manual measurements": df})
    w.set_plot_type("Histogram")
    w.set_normalization("density")
    return w


def _note(window) -> str:
    return next(ln.strip() for ln in window.details_text().splitlines() if "bin width" in ln)


@pytest.mark.parametrize("col", ["amplitude", "var"])
def test_an_undeclared_unit_is_not_called_dimensionless(window, col):
    window.set_histogram_column(col)
    note = _note(window)
    assert "dimensionless" not in note, note
    assert f"the units of {col}" in note and "does not declare" in note, note


def test_a_declared_dimensionless_column_still_gets_the_dimensionless_sentence(window):
    """``frac_depth`` is a ratio; ``1/`` on it would be nonsense, and the sentence that
    says so is the one S6-03 wrote."""
    window.set_histogram_column("frac_depth")
    assert "a dimensionless density" in _note(window)


def test_a_declared_unit_still_names_its_density_unit(window):
    window.set_histogram_column("depth_pA")
    assert "bin width in pA — a density in 1/pA" in _note(window)
