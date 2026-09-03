"""S3 EXPL3-03 / EXPL3-04 — a seconds-valued column is not automatically a dwell.

THE DEFECT, one gate. ``PlotsWindow._overlay_fit`` decided whether to tell the dwell
fitters what they were fitting with

    dwell_kw = {} if _u == "s" else {"quantity": _q, "unit": _u}

and ``_column_quantity`` answers ``"s"`` for EVERY ``*_s`` column. So ``rise_s``,
``fall_s`` and ``t_start_s`` -- seconds, but not dwells -- were handed to the fitters
under the dwell contract, and every sentence written about them spoke about a different
column:

* refusals named the wrong column: "Fit unavailable: dwell_s has only 5 events" (and,
  for the peaked families, bare "dwell") three lines under "Histogram of rise_s";
* the fitter's guessed-``t_min`` warning printed a SECOND copy of the panel's own note,
  in dwell words, and both halves were false -- "provenance has no 'min_dwell_s'" on a
  recording whose provenance declares one, and the shortest observed FALL time called
  "the shortest observed dwell", with "pass an EventSet from detect() to get it right"
  advice the user had already followed;
* a user-typed Fit range max was attributed to the detector: "events longer than it were
  never reported (detector max_dwell_s, or curation)" -- on a run whose real
  ``max_dwell_s`` was 100x larger.

The window already owns the list of columns the dwell provenance describes
(``_DWELL_COLUMNS``, the same list ``_dwell_inputs`` refuses on, with the comment "a
histogram of rise_s or ecd_pA_s is not a dwell distribution"), so the gate now asks it.
``depth_pA`` was already correct; these tests pin that ``rise_s`` and ``fall_s`` are
treated the way ``depth_pA`` is, and that ``dwell_s`` keeps its historical wording
bit-for-bit.
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

PROV = {"min_dwell_s": 2e-5, "max_dwell_s": 5e-2, "fs": 50_000.0}


# ================================================================================================
# builders
# ================================================================================================

def _frame(n: int, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": rng.exponential(2e-3, n) + 2e-5,
        "rise_s": np.abs(rng.normal(8e-5, 2e-5, n)) + 4e-5,
        "fall_s": np.abs(rng.normal(9e-5, 2e-5, n)) + 4e-5,
        "depth_pA": np.abs(rng.normal(80.0, 12.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _window(qtbot, col: str, fit: str, *, n: int = 400, hi: str | None = None) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run": _frame(n)}, {"run": dict(PROV)})
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    w.set_time_unit("s")
    w.set_fit(fit)
    if hi is not None:
        w._fit_hi.setText(hi)
        w._changed()
    return w


DWELL_FAMILIES = ["exponential", "gamma (peaked)", "inverse-gaussian (first passage)"]


# ================================================================================================
# a refusal names the column the user actually fitted
# ================================================================================================

@pytest.mark.parametrize("col", ["rise_s", "fall_s"])
@pytest.mark.parametrize("fit", DWELL_FAMILIES)
def test_a_refusal_names_the_fitted_column_not_dwell(qapp, qtbot, col, fit):
    w = _window(qtbot, col, fit, n=5)     # below stats._MIN_EVENTS: it MUST refuse
    line = next((ln for ln in w.details_text().splitlines() if "Fit unavailable" in ln), "")
    assert line, w.details_text()
    assert col in line, line
    assert "dwell" not in line, line


@pytest.mark.parametrize("fit", DWELL_FAMILIES)
def test_a_dwell_refusal_still_says_dwell(qapp, qtbot, fit):
    """The control: ``dwell_s`` IS a dwell and keeps the historical wording."""
    w = _window(qtbot, "dwell_s", fit, n=5)
    line = next((ln for ln in w.details_text().splitlines() if "Fit unavailable" in ln), "")
    assert line, w.details_text()
    assert "dwell" in line, line


# ================================================================================================
# one fact, one line -- and the line is the true one
# ================================================================================================

@pytest.mark.parametrize("col", ["rise_s", "fall_s"])
def test_the_dwell_provenance_story_is_not_repeated_in_dwell_words(qapp, qtbot, col):
    text = _window(qtbot, col, "exponential").details_text()
    assert "provenance has no 'min_dwell_s'" not in text, text
    assert "shortest observed dwell" not in text, text
    # ...and the panel's own note, in the column's own terms, is still there.
    assert f"no detection limit is known for “{col}”" in text, text


@pytest.mark.parametrize("col", ["rise_s", "fall_s"])
def test_the_provenance_is_not_denied_when_the_recording_declares_it(qapp, qtbot, col):
    """The same window on ``dwell_s`` reads min_dwell_s out of the SAME provenance."""
    w = _window(qtbot, "dwell_s", "exponential")
    assert "detection limit t_min = 2e-05 s (from min_dwell_s)" in w.details_text()
    assert "provenance has no" not in _window(qtbot, col, "exponential").details_text()


@pytest.mark.parametrize("col", ["rise_s", "fall_s"])
def test_a_typed_ceiling_is_not_blamed_on_the_detector(qapp, qtbot, col):
    """PROV declares max_dwell_s = 0.05 s; a 4e-4 s ceiling cannot have come from it."""
    text = _window(qtbot, col, "exponential", hi="4e-4").details_text()
    line = next((ln for ln in text.splitlines() if "right truncation at" in ln), "")
    assert line, text
    assert "detector max_dwell_s" not in line, line
    # The hedge "(a fit range, or curation)" was this window's best answer while it did
    # not tell the fitter WHICH; it now does (`_t_max_source`), so the pane names the
    # typed range outright. The guarantee under test -- the detector is not blamed -- is
    # unchanged and stronger.
    assert "NOT a detector limit" in line, line
    assert col in line, line


def test_a_picoamp_ceiling_is_worded_the_same_way_it_always_was(qapp, qtbot):
    """The control that was already right, and is now specific rather than hedged."""
    w = _window(qtbot, "depth_pA", "exponential", hi="120")
    line = next((ln for ln in w.details_text().splitlines() if "right truncation at" in ln), "")
    assert line, w.details_text()
    assert "depth_pA values above it" in line and "NOT a detector limit" in line, line
    assert "detector max_dwell_s" not in line, line


def test_a_dwell_ceiling_still_names_the_detector(qapp, qtbot):
    """A dwell ceiling really can come from ``max_dwell_s``; that wording stays.

    CORRECTED. This test used to type ``hi="4e-3"`` and still demand the words
    "detector max_dwell_s" -- i.e. it pinned the fault it was written to guard, on the
    one column EXPL3-04 deliberately froze. ``PROV`` declares ``max_dwell_s = 0.05 s``,
    so a 0.004 s ceiling cannot be the detector's; the case where the detector's ceiling
    IS the modelled bound is the one with nothing typed, and that is what this control
    now builds. The typed-ceiling case is the test directly below.
    """
    w = _window(qtbot, "dwell_s", "exponential")
    line = next((ln for ln in w.details_text().splitlines() if "right truncation at" in ln), "")
    assert line, w.details_text()
    assert f"t_max={PROV['max_dwell_s']:g} s" in line, line
    assert "detector max_dwell_s" in line, line


def test_a_typed_dwell_ceiling_is_not_blamed_on_the_detector_either(qapp, qtbot):
    """The same rule EXPL3-04 restored for ``rise_s``/``fall_s``, on ``dwell_s``.

    A dwell column is the one place a detector ceiling is even possible, which is
    exactly why the wrong story survived here: 0.004 s is a plausible ``max_dwell_s``.
    It is not this run's -- ``PROV`` says 0.05 s -- and the pane must not credit a
    number it never printed for a truncation the analyst imposed.
    """
    w = _window(qtbot, "dwell_s", "exponential", hi="4e-3")
    line = next((ln for ln in w.details_text().splitlines() if "right truncation at" in ln), "")
    assert line, w.details_text()
    assert "t_max=0.004 s" in line, line
    assert "detector max_dwell_s" not in line, line
    assert "NOT a detector limit" in line, line


# ================================================================================================
# the fit itself is unchanged: only the words move
# ================================================================================================

def test_the_fitted_numbers_are_labelled_with_the_fitted_column(qapp, qtbot):
    w = _window(qtbot, "rise_s", "exponential")
    fit = w.fit_result()
    assert fit is not None
    assert fit.meta["quantity"] == "rise_s"
    assert "rise_s" in fit.model
    # the parameter is still a time in seconds -- only the naming moved
    assert fit.meta["param_units"]["tau_1"] == "s"


def test_a_dwell_fit_keeps_its_anonymous_dwell_model_name(qapp, qtbot):
    fit = _window(qtbot, "dwell_s", "exponential").fit_result()
    assert fit.meta["quantity"] is None
    assert fit.model == "exponential (doubly-truncated MLE)"   # PROV bounds both sides
    assert " on " not in fit.model
