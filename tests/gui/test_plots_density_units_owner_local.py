"""Manual/local-only historical owner-data density check; excluded from automated gates.

Explicit opt-in: NANOSCOPE_RUN_OWNER_DATA=1. No fixture substitutes for the measured
982-event file, and the deterministic sibling never reads this owner path.
"""
import os
import pathlib
import numpy as np
import pandas as pd
import pytest
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow
from .test_plots_density_units import _drawn_area, _stored_area, _y_label

#: The owner's own detected events. Present on his machine, absent on CI — so the numeric
#: pins measured off it live in ONE test that skips, and every structural assertion below
#: runs on the synthetic frame instead. A guard that only ever runs on one laptop is not a
#: guard.
REAL_EVENTS = pathlib.Path(
    "/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv")

# 5. the owner's own file — the numbers the report quotes, pinned where they exist
# ================================================================================================

@pytest.mark.skipif(os.environ.get("NANOSCOPE_RUN_OWNER_DATA") != "1" or not REAL_EVENTS.exists(),
                    reason="manual owner-data check requires explicit local opt-in and exact file")
def test_the_measured_numbers_on_the_real_982_event_run(qapp, qtbot):
    """Every figure report S6-03 quotes, re-measured through the shipped window.

    Kept apart from the rest of the file, and skipped rather than approximated, because a
    pin on real data is only meaningful against THAT data.
    """
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_data({"−700 mV": pd.read_csv(REAL_EVENTS)})
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_bins(30)
    window.set_fit("none")
    window.set_normalization("density")

    window._run_plot()
    bottom, left = window._plot.getAxis('bottom'), window._plot.getAxis('left')
    window.set_log_x(False)
    item = window._hist_items[0]
    assert float(np.max(item.yData)) == pytest.approx(160.876232, abs=1e-6)
    assert _drawn_area(item, bottom, left) == pytest.approx(1., rel=1e-9)
    assert _stored_area(item) == pytest.approx(1.0, rel=1e-12)
    assert _y_label(window) == "density (1/ms)"
    window.set_log_x(True)
    item = window._hist_items[0]
    assert float(np.max(item.yData)) == pytest.approx(154.195604, abs=1e-6)
    assert _drawn_area(item, bottom, left) == pytest.approx(23.3011485 / 1000., rel=1e-8)
    assert _stored_area(item) == pytest.approx(1.0, rel=1e-12)
    assert _y_label(window) == "density (1/ms)"
