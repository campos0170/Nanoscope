"""The Plots window must draw the detection the Analysis panel is showing.

A channel can hold several detections at once (``ResultStore`` is keyed
``(doc_id, channel, module_id)``), so "the current recording" stopped identifying a
table the moment discovery started offering one source per detector.

MEASURED on the owner's -1000 mV recording on 2026-09-02, driven through the real
MainWindow at 1512x900: Schmitt (110 events) and CUSUM (250) both stored, the panel
showing CUSUM, and Plots opening with Schmitt's 110 rows ticked, drawn and captioned
"(n = 110)" under the bare filename -- a figure of a table the user was not looking
at, with nothing on it that says so.

These pin the window half of the fix: which row opens ticked, that a row says which
detector made it, and that ticking both really does draw a two-detector comparison
(the "Next"-list item this delivers for free, since the window has been a multi-source
overlay engine all along).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.analysis.events import SCHEMA
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow

FS = 50_000.0
DOC = "doc-1"


def _frame(n: int, *, tau: float = 3e-4, t_min: float = 5e-5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for name, (dtype, _u, _d) in SCHEMA.items():
        if dtype == "int64":
            cols[name] = np.arange(n, dtype=np.int64)
        elif dtype == "bool":
            cols[name] = np.zeros(n, dtype=bool)
        elif dtype == "string":
            cols[name] = pd.Series([""] * n, dtype="string")
        else:
            cols[name] = np.zeros(n, dtype=np.float64)
    cols["dwell_s"] = t_min + rng.exponential(tau, n)
    cols["baseline_pA"] = np.full(n, 120.0)
    cols["depth_pA"] = np.abs(rng.normal(30.0, 3.0, n))
    cols["frac_depth"] = cols["depth_pA"] / 120.0
    cols["accepted"] = np.ones(n, dtype=bool)
    return pd.DataFrame(cols)


def _detection(module_id: str, *, n: int, label: str, colour: str,
               current: bool = False, seed: int = 0) -> PlotSource:
    """One source as ``discover_plot_sources`` builds it: per-detector key, per-detector
    label, the channel shared with its sibling."""
    key = "rec#0" if module_id == "detect.schmitt" else f"rec#0@{module_id}"
    return PlotSource(
        key=key, doc_key="rec", label=label, df=_frame(n, seed=seed),
        provenance={"fs": FS, "min_dwell_s": 5e-5,
                    "baseline_fingerprint": "sha256:0000aaaa1111"},
        colour=colour, doc_id=DOC, channel=0, module_id=module_id,
        display_name="apo.abf", calibration_fingerprint=(1.0, 0.0), current=current,
    )


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


@pytest.fixture
def two_detectors():
    """Schmitt and CUSUM on ONE channel. Schmitt carries ``current=True`` because a
    caller that names no module flags the channel's FIRST detection -- the guess this
    window has to be able to overrule."""
    return [
        _detection("detect.schmitt", n=110, label="Schmitt · apo.abf",
                   colour="#0072B2", current=True, seed=1),
        _detection("detect.cusum", n=250, label="CUSUM · apo.abf",
                   colour="#D55E00", seed=2),
    ]


def test_the_window_opens_ticking_the_detection_the_panel_is_showing(window, two_detectors):
    window.set_active_detection((DOC, 0, "detect.cusum"))
    window.set_data(two_detectors)

    assert window.selected_source_keys() == ["rec#0@detect.cusum"]
    assert window.primary_source().n_rows == 250, (
        "the panel is showing CUSUM's 250 rows, so the figure must be of those")


def test_the_window_offers_both_detectors_as_separate_rows(window, two_detectors):
    window.set_data(two_detectors)

    assert window.source_labels() == ["Schmitt · apo.abf", "CUSUM · apo.abf"]
    assert window._source_list.count() == 2


def test_a_source_row_says_which_detector_made_its_rows(window, two_detectors):
    window.set_data(two_detectors)
    tips = [window._source_list.item(i).toolTip() for i in range(2)]

    assert "Schmitt (detect.schmitt)" in tips[0]
    assert "CUSUM (detect.cusum)" in tips[1]
    assert "110 rows" in tips[0] and "250 rows" in tips[1]


def test_ticking_both_detectors_draws_a_two_detector_comparison(window, two_detectors):
    """The comparison surface the owner's "Next" list asks for, with no new window: the
    overlay engine was already here, it was the discovery that offered it one table."""
    window.set_data(two_detectors)
    window.set_sources(["rec#0", "rec#0@detect.cusum"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_fit("none")

    assert len(window._hist_items) == 2, "one histogram per detector"
    legend = window.legend_labels()
    assert any("Schmitt" in t for t in legend), legend
    assert any("CUSUM" in t for t in legend), legend
    assert len({s.colour for s in window.selected_sources()}) == 2


def test_without_an_active_detection_the_current_flag_still_decides(window, two_detectors):
    """The fallback, pinned: a window with no owner to ask (every bare-constructed one,
    and the notebook-style callers) keeps exactly the pre-existing behaviour."""
    window.set_data(two_detectors)

    assert window.selected_source_keys() == ["rec#0"]


def test_a_refresh_does_not_move_the_user_off_the_source_they_ticked(window, two_detectors):
    """The active detection chooses the FIRST push only. Re-running a detector re-pushes
    the sources, and silently swapping the drawn table underneath a user who ticked one
    on purpose is the same class of surprise this fix exists to remove."""
    window.set_active_detection((DOC, 0, "detect.cusum"))
    window.set_data(two_detectors)
    window.set_sources(["rec#0"])

    window.set_data(two_detectors)          # what MainWindow's Refresh does

    assert window.selected_source_keys() == ["rec#0"]
