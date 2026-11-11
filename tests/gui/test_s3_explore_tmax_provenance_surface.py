"""S3 (repair) — the Details pane names the bound the likelihood actually got, and WHO set it.

``_restrict`` intersects the detector's ceiling with the Fit range the analyst typed, and
hands the intersection to the fitter. The FIT RANGE line was taught to print that
intersection (EXPL3-01); the MODELLED-truncation line one row above it was not taught who
the surviving bound belongs to, and asserted the detector's story unconditionally on
``dwell_s``:

    right truncation at t_max=0.004 s is MODELLED: events longer than it were never
    reported (detector max_dwell_s, or curation)

on a run whose ``max_dwell_s`` is 0.05 s and appears nowhere on the pane. Two sentences
about one fit, one of them false -- the same fault EXPL3-04 filed for ``rise_s``, on the
one column it deliberately froze bit-for-bit.

``PlotsWindow._t_max_source`` is the mirror of the ``t_min_source`` half of
``_as_fitted``: it answers which bound survived the intersection, and the fitter words it
through ``stats.truncation_source_clause``. These tests read the rendered Details pane of
a real ``PlotsWindow``.

2026-11-11: the window is no longer put in seconds. Since 8661d35 the Explore window's
Time unit holds only "ms" (disabled), so ``dwell_s`` is shown in ms and its Fit range
boxes are read in ms. The constants below stay in SECONDS (the unit the fitters, the
provenance and ``fit.meta`` speak); ``_window`` types a ceiling the way a person now does,
and the Details lines are quoted in ms. The quoted 0.004 s above is the defect as it was
filed, in the unit it was filed in.
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

DET_MIN_S = 2.0e-4
DET_MAX_S = 5.0e-2
N = 900


def _frame(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dwell = np.clip(rng.exponential(1.5e-3, N) + DET_MIN_S, DET_MIN_S, DET_MAX_S)
    return pd.DataFrame({
        "dwell_s": dwell,
        "rise_s": np.abs(rng.normal(8e-5, 2e-5, N)) + 4e-5,
        "depth_pA": np.abs(rng.normal(80.0, 12.0, N)),
        "accepted": np.ones(N, dtype=bool),
        "filter_limited": np.zeros(N, dtype=bool),
    })


#: The declared time columns, which the Explore window shows -- and reads its Fit range
#: boxes in -- in ms. Spelled out here on purpose (not imported from the window).
_MS_COLUMNS = ("dwell_s", "rise_s", "fall_s")


def _typed_ceiling(col: str, hi: str) -> str:
    """What a person types into the Fit range max box for the ceiling ``hi``.

    ``hi`` is written as these tests always wrote it: SECONDS for a time column. The box is
    read in the unit the axis shows, so "4e-3" s is typed "4" (ms); typing "4e-3" now would
    mean 0.004 ms and empty the fit instead of exercising the note under test.
    """
    return f"{float(hi) * 1e3:g}" if col in _MS_COLUMNS else hi


def _window(qtbot, col: str, *, hi: str | None = None, provenance: bool = True) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    prov = {"min_dwell_s": DET_MIN_S, "max_dwell_s": DET_MAX_S, "fs": 50_000.0}
    w.set_data({"run": _frame()}, {"run": prov} if provenance else None)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    # 2026-11-11: this used to call ``w.set_time_unit("s")``, which now raises -- the combo
    # offers only "ms" since 8661d35. The unit is pinned instead of set: every ceiling typed
    # and every "t_max=... ms" quoted below depends on it.
    assert w.time_unit() == "ms"
    w.set_fit("exponential")
    if hi is not None:
        w._fit_hi.setText(_typed_ceiling(col, hi))
        w._changed()
    return w


def _modelled(w: PlotsWindow) -> str:
    line = next((ln for ln in w.details_text().splitlines() if "right truncation at" in ln), "")
    assert line, w.details_text()
    return line


def _fit_range(w: PlotsWindow) -> str:
    line = next((ln for ln in w.details_text().splitlines() if "FIT RANGE" in ln), "")
    assert line, w.details_text()
    return line


# ================================================================================================
# who imposed the ceiling
# ================================================================================================

def test_the_detector_ceiling_is_credited_to_the_detector(qapp, qtbot):
    """Nothing typed: ``max_dwell_s`` IS the modelled bound, and is named."""
    w = _window(qtbot, "dwell_s")
    line = _modelled(w)
    assert f"t_max={DET_MAX_S * 1e3:g} ms" in line, line      # 0.05 s, read in ms
    assert "detector max_dwell_s" in line, line
    assert w.fit_result().meta["t_max_source"] == stats.T_MAX_FROM_DETECTOR


def test_a_typed_ceiling_below_the_detectors_is_not_credited_to_it(qapp, qtbot):
    w = _window(qtbot, "dwell_s", hi="4e-3")
    line = _modelled(w)
    assert "t_max=4 ms" in line, line
    assert "max_dwell_s" not in line, line
    assert "NOT a detector limit" in line, line
    assert w.fit_result().meta["t_max_source"] == stats.T_MAX_FROM_FIT_RANGE
    # 2026-11-11: the box was typed in ms and reached the likelihood in seconds.
    assert w.fit_result().meta["t_max"] == pytest.approx(4e-3)


def test_a_typed_ceiling_above_the_detectors_loses_and_says_so(qapp, qtbot):
    """The intersection keeps the detector's bound, so the detector's story is the true one."""
    w = _window(qtbot, "dwell_s", hi="9.0")
    line = _modelled(w)
    assert f"t_max={DET_MAX_S * 1e3:g} ms" in line, line      # typed 9 s lost to 50 ms
    assert "detector max_dwell_s" in line, line
    assert w.fit_result().meta["t_max_source"] == stats.T_MAX_FROM_DETECTOR


def test_a_ceiling_with_no_provenance_behind_it_is_the_typed_one(qapp, qtbot):
    """No detection provenance at all: there is no detector limit to credit."""
    w = _window(qtbot, "dwell_s", hi="4e-3", provenance=False)
    line = _modelled(w)
    assert "max_dwell_s" not in line, line
    assert "NOT a detector limit" in line, line


def test_a_non_dwell_column_still_refuses_the_detector_story(qapp, qtbot):
    """CONTROL — EXPL3-04's columns keep their answer; ``rise_s`` has no dwell ceiling."""
    line = _modelled(_window(qtbot, "rise_s", hi="4e-4"))
    assert "max_dwell_s" not in line, line
    assert "rise_s values above it" in line, line


# ================================================================================================
# one pane, one truth
# ================================================================================================

@pytest.mark.parametrize("hi", [None, "4e-3", "9.0"])
def test_the_two_lines_quote_the_same_bound(qapp, qtbot, hi):
    """The MODELLED line and the FIT RANGE line describe ONE fit, so they agree."""
    w = _window(qtbot, "dwell_s", hi=hi)
    fitted_hi = float(w.fit_result().meta["t_max"])      # seconds: what the likelihood got
    assert f"t_max={fitted_hi * 1e3:g} ms" in _modelled(w), _modelled(w)
    if hi is not None:
        assert f"{fitted_hi * 1e3:.12g}] ms" in _fit_range(w), _fit_range(w)


def test_the_label_does_not_move_the_numbers(qapp, qtbot):
    """Only the words change. The window's fit must still equal the bare fitter's over
    the same sample and the same window -- the provenance token is not an input to the
    likelihood."""
    w = _window(qtbot, "dwell_s", hi="4e-3")
    fit = w.fit_result()
    dwell = _frame()["dwell_s"].to_numpy()
    direct = stats.fit_dwell_exponential(dwell[dwell <= 4e-3], t_min=DET_MIN_S,
                                         t_max=4e-3, n_boot=0)
    assert fit.n == direct.n
    assert fit.params["tau_1"] == pytest.approx(direct.params["tau_1"], rel=1e-9)
