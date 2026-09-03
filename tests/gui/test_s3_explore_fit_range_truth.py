"""S3 EXPL3-01 — the FIT RANGE line reports the window the likelihood actually got.

THE DEFECT, both facets, one root: the pane described the fit by echoing the CONTROLS
rather than by reading the fit.

* ``_restrict``'s own docstring says the truncation is "the INTERSECTION of what the
  detector could report and what the user asked to fit", and that intersected pair is
  what every branch hands the fitter. The FIT RANGE note was built from the RAW typed
  pair, so with provenance narrowing a wide typed window the SAME pane stated two
  different truncations for one fit: "FIT RANGE [1e-06, 5] s ... the likelihood is
  TRUNCATED to this window" beside a censoring string reading
  ``t_min=2e-05 <= t <= t_max=0.0317424`` -- the printed lower bound 20x too small and
  the upper 158x too large, on the one line whose job is to say what the reported tau
  is conditioned on.
* ``modules.stamp_fit_provenance`` assigns ``meta['t_min']`` / ``meta['t_min_source']``
  UNCONDITIONALLY (every other key in it goes through ``setdefault``). That is right for
  the Fits panel, whose resolution IS the fitted window, and wrong here: a fit
  conditioned on a typed floor of 0.001 s came back stamped 2e-05 s "from min_dwell_s",
  and ``plot.figures._fit_caveats`` printed THAT onto an exported figure as the
  detection limit. One FitResult, two floors.

THE RULE THESE PIN. Every sentence the pane prints about a fit describes the fit that
actually ran -- the window it was conditioned on, in the fitted column's unit, once.
Ground truth here is taken from the FitResult and from the fitter's own censoring
string, never from another string the same code wrote.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

#: A detector window narrower than the range the tests type, on BOTH sides.
DET_MIN_S = 2.0e-4
DET_MAX_S = 5.0e-3


# ================================================================================================
# builders
# ================================================================================================

def _frame(n: int = 1200, *, tau: float = 1.5e-3, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dwell = rng.exponential(tau, n) + DET_MIN_S
    dwell = np.clip(dwell, DET_MIN_S, DET_MAX_S)
    return pd.DataFrame({
        "dwell_s": dwell,
        "depth_pA": np.abs(rng.normal(80.0, 15.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _window(qtbot, *, col="dwell_s", fit="exponential", lo=None, hi=None,
            provenance=True, unit="s") -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    prov = {"min_dwell_s": DET_MIN_S, "max_dwell_s": DET_MAX_S, "fs": 50_000.0}
    w.set_data({"run": _frame()}, {"run": prov} if provenance else None)
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    w.set_time_unit(unit)
    w.set_fit(fit)
    if lo is not None:
        w._fit_lo.setText(str(lo))
    if hi is not None:
        w._fit_hi.setText(str(hi))
    if lo is not None or hi is not None:
        w._changed()
    return w


def _fit_range_line(w: PlotsWindow) -> str:
    line = next((ln for ln in w.details_text().splitlines() if "FIT RANGE" in ln), "")
    assert line, "a typed fit range must produce a FIT RANGE line\n" + w.details_text()
    return line


def _bounds(line: str) -> tuple[float, float]:
    m = re.search(r"FIT RANGE \[([^,]+), ([^\]]+)\]", line)
    assert m, line
    return float(m.group(1)), float(m.group(2))


# ================================================================================================
# facet 1 -- the printed window IS the fitted window
# ================================================================================================

@pytest.mark.parametrize("fit", ["exponential", "gamma (peaked)",
                                 "inverse-gaussian (first passage)"])
def test_fit_range_prints_the_intersection_not_the_typed_pair(qapp, qtbot, fit):
    """Type a window WIDER than the detector's on both sides: the detector's wins."""
    w = _window(qtbot, fit=fit, lo="1e-9", hi="9.0")
    result = w.fit_result()
    assert result is not None
    shown_lo, shown_hi = _bounds(_fit_range_line(w))
    # Round-trips to the fit's OWN censoring bounds, which is the whole point: this
    # line is copied into a methods section beside those numbers.
    assert shown_lo == pytest.approx(float(result.meta["t_min"]), rel=0, abs=1e-12)
    assert shown_hi == pytest.approx(float(result.meta["t_max"]), rel=0, abs=1e-12)
    assert shown_lo == pytest.approx(DET_MIN_S, rel=1e-9)
    assert shown_hi == pytest.approx(DET_MAX_S, rel=1e-9)


def test_the_pane_never_states_two_truncations_for_one_fit(qapp, qtbot):
    """The FIT RANGE bounds and the censoring string must agree, in one pane."""
    w = _window(qtbot, lo="1e-9", hi="9.0")
    result = w.fit_result()
    censoring = str(result.meta.get("censoring", ""))
    shown_lo, shown_hi = _bounds(_fit_range_line(w))
    nums = [float(x) for x in re.findall(r"t_m(?:in|ax)=([0-9.eE+-]+)", censoring)]
    assert nums, censoring
    assert shown_lo == pytest.approx(nums[0], rel=1e-9)
    assert shown_hi == pytest.approx(nums[-1], rel=1e-9)


def test_a_narrowed_range_says_so_rather_than_silently_swapping_numbers(qapp, qtbot):
    """A refusal (and a correction) names its real cause: the typed pair is quoted."""
    line = _fit_range_line(_window(qtbot, lo="1e-9", hi="9.0"))
    assert "typed range" in line, line
    assert "1e-09" in line and "9" in line, line
    assert "intersected" in line, line


def test_an_unnarrowed_range_prints_no_correction_clause(qapp, qtbot):
    """When the typed window IS the fitted window the line stays as it was."""
    line = _fit_range_line(_window(qtbot, lo="0.001", hi="0.004"))
    assert "typed range" not in line, line
    lo, hi = _bounds(line)
    assert (lo, hi) == pytest.approx((0.001, 0.004))


def test_fit_range_prints_in_the_axis_unit(qapp, qtbot):
    """The unit rule the earlier sweeps installed is untouched by the window rule.

    The boxes are read in the axis's own unit, so 1-4 ms is the same window as the
    0.001-0.004 s typed above; it must come back out labelled ms.
    """
    w = _window(qtbot, lo="1", hi="4", unit="ms")
    line = _fit_range_line(w)
    assert "] ms —" in line, line
    lo, hi = _bounds(line)
    assert (lo, hi) == pytest.approx((1.0, 4.0))


def test_the_gaussian_mixture_says_cropped_because_it_does_not_truncate(qapp, qtbot):
    """``fit_gaussian_mixture`` is fitted to the surviving values, not conditioned on
    the window -- so the sentence that says "TRUNCATED, not merely cropped" is exactly
    backwards there."""
    w = _window(qtbot, col="depth_pA", fit="gaussian-mixture", lo="40", hi="120")
    line = _fit_range_line(w)
    assert "CROPPED" in line, line
    assert "TRUNCATED to this window" not in line, line


# ================================================================================================
# facet 2 -- the provenance stamp describes the fit, not the resolution
# ================================================================================================

def test_a_typed_floor_above_the_detectors_is_the_floor_the_fit_is_stamped_with(qapp, qtbot):
    w = _window(qtbot, lo="0.001")
    result = w.fit_result()
    assert float(result.meta["t_min"]) == pytest.approx(0.001, rel=1e-12)
    assert result.meta["t_min_source"] == "explicit"
    assert "t_min=0.001" in str(result.meta.get("censoring", ""))


def test_the_detectors_floor_is_still_named_when_it_is_the_one_that_survived(qapp, qtbot):
    """The provenance stamp is corrected only where it would be false."""
    w = _window(qtbot, lo="1e-9")
    result = w.fit_result()
    assert float(result.meta["t_min"]) == pytest.approx(DET_MIN_S, rel=1e-12)
    assert result.meta["t_min_source"] == "min_dwell_s"


def test_a_typed_ceiling_below_the_detectors_is_the_ceiling_that_is_stamped(qapp, qtbot):
    w = _window(qtbot, hi="0.002")
    result = w.fit_result()
    assert float(result.meta["t_max"]) == pytest.approx(0.002, rel=1e-12)


def test_no_fit_range_leaves_the_provenance_stamp_bitforbit(qapp, qtbot):
    """The untyped case must be exactly what it always was."""
    w = _window(qtbot)
    result = w.fit_result()
    assert float(result.meta["t_min"]) == pytest.approx(DET_MIN_S, rel=1e-12)
    assert result.meta["t_min_source"] == "min_dwell_s"
    assert "FIT RANGE" not in w.details_text()
