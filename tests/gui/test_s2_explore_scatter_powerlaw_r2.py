"""S2 EXPL2-6 / EXPL2-7 — the scatter fit line survives a zero x, and R² = nan says why.

* EXPL2-6: ``populations.linear_fit(log_log=True)`` drops non-positive pairs by
  design and fits the rest — but the Plots scatter handed ``line_points`` the
  min/max over ALL plotted x, so one x == 0 on linear axes produced
  "Fit unavailable: log-log line_points needs xmin, xmax > 0; got (0.0, 3.0)":
  a false refusal (the fit succeeded, n = 39) quoting an internal helper name.
  The line now spans the positive points the fit actually used, and the exclusion
  is stated in user terms.

* EXPL2-7: ``linear_fit`` documents ``r2 = nan`` for a zero-variance response, and
  two surfaces printed the NaN bare — the Plots scatter caption
  ("y = ...·x + 5   R² = nan   (n = 40)") and the voltage-dependence Goodness line
  ("Goodness of fit:  R² = nan   ·   n = 2 recording(s)") — so it read as a
  computation failure. Both now say the cause: the response does not vary, so a
  line explains nothing (the PSD panel's ``_fmt_q`` rule — never let a non-finite
  value reach a bare format spec).

All assertions go through the real widgets: ``PlotsWindow`` public setters and a
``VDepWindow`` over a real two-document Session.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import tempfile
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow


def _scatter_window(qtbot, df: pd.DataFrame) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"probe": df})
    w.set_plot_type("Scatter")
    w.set_scatter_columns("xcol", "ycol")
    return w


def _df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"xcol": x, "ycol": y,
                         "accepted": np.ones(x.size, dtype=bool),
                         "filter_limited": np.zeros(x.size, dtype=bool)})


# ================================================================================================
# EXPL2-6 — power law with one x == 0 on linear axes
# ================================================================================================

def test_power_law_with_a_zero_x_draws_the_line_over_the_positive_points(qapp, qtbot):
    x = np.linspace(0.0, 3.0, 40)                     # x[0] == 0.0
    y = 2.0 * np.maximum(x, 1e-9) ** 1.5 + 0.1
    w = _scatter_window(qtbot, _df(x, y))
    w.set_scatter_fit("power law (log-log)")
    text = w.details_text()

    # The fit is available — it was always available — and the line is drawn.
    assert w._scatter_fit_item is not None, text
    assert "Fit unavailable" not in text, text
    # No internal helper names on a user surface.
    assert "line_points" not in text, text
    # The exclusion is disclosed, in user terms, with the count.
    assert "1 of 40 point(s)" in text, text
    assert "excluded from the fit" in text, text
    # ...and the line spans exactly the positive points the fitter used.
    lx = w._scatter_fit_item.getData()[0]
    assert float(np.min(lx)) == pytest.approx(float(x[1]))
    assert float(np.max(lx)) == pytest.approx(3.0)
    assert "(n = 39)" in text, text


def test_power_law_on_all_positive_data_says_nothing_extra(qapp, qtbot):
    """CONTROL — no exclusions, no exclusion note; the historical caption stands."""
    x = np.linspace(1.0, 10.0, 40)
    w = _scatter_window(qtbot, _df(x, 3.0 * x ** 2))
    w.set_scatter_fit("power law (log-log)")
    text = w.details_text()
    assert "excluded from the fit" not in text, text
    assert "y = 3·x^2" in text and "R² = 1.000" in text, text


def test_power_law_with_no_positive_pairs_still_refuses(qapp, qtbot):
    """CONTROL — a refusal that is TRUE stays a refusal: with no positive pairs
    there is genuinely no fit, and the scatter itself still stands."""
    x = np.linspace(-5.0, 0.0, 20)
    w = _scatter_window(qtbot, _df(x, np.abs(x) + 1.0))
    w.set_scatter_fit("power law (log-log)")
    text = w.details_text()
    assert "Fit unavailable" in text, text
    assert w._scatter_fit_item is None


# ================================================================================================
# EXPL2-7a — the Plots scatter caption on a zero-variance response
# ================================================================================================

def test_scatter_flat_response_explains_undefined_r2(qapp, qtbot):
    w = _scatter_window(qtbot, _df(np.linspace(1.0, 10.0, 40), np.full(40, 5.0)))
    w.set_scatter_fit("linear")
    text = w.details_text()
    title = w.plot_item().titleLabel.text
    for surface in (text, title):
        assert "R² = nan" not in surface, surface
        assert "R² undefined (y does not vary)" in surface, surface
    # Details additionally say WHY there is nothing to report.
    assert "no variance" in text and "a line explains nothing" in text, text


def test_scatter_sloped_response_keeps_its_numeric_r2(qapp, qtbot):
    """CONTROL — a real trend still prints the number, exactly as before."""
    x = np.linspace(1.0, 10.0, 40)
    w = _scatter_window(qtbot, _df(x, 2.0 * x + 1.0))
    w.set_scatter_fit("linear")
    text = w.details_text()
    assert "y = 2·x + 1" in text and "R² = 1.000" in text, text
    assert "undefined" not in text, text


# ================================================================================================
# EXPL2-7b — the voltage-dependence Goodness line on identical capture rates
# ================================================================================================

def _current_doc(tmp: Path, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _events_df(n: int) -> pd.DataFrame:
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": 2e-3,
            "baseline_pA": -1000.0, "depth_pA": 60.0,
            "frac_depth": 0.1, "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _vdep_session(counts: tuple[int, int]) -> Session:
    tmp = Path(tempfile.mkdtemp(prefix="s2-expl-vdep-"))
    session = Session()
    for (mv, name), n_ev in zip(((600.0, "p.dat"), (-600.0, "m.dat")), counts):
        did = session.add_document(_current_doc(tmp, mv, name))
        ev = EventSet(_events_df(n_ev), {"analyzed_duration_s": 10.0, "recording": "rec"})
        session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=did, channel=0,
            params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))
    return session


def test_vdep_identical_rates_explain_undefined_r2(qapp, qtbot):
    w = VDepWindow(_vdep_session((40, 40)))       # same capture rate at both voltages
    qtbot.addWidget(w)
    w.set_fit("linear")
    text = w.details_text()
    assert "R² = nan" not in text, text
    good = next(ln for ln in text.splitlines() if ln.startswith("Goodness of fit"))
    assert "R² undefined" in good, good
    assert "explains nothing" in good, good
    assert "n = 2 recording(s)" in good, good
    # The title carries the same honest verdict in its short form (its wrap
    # budget is real), not a bare nan; the Goodness line above holds the cause.
    title = w._plot.titleLabel.text
    assert "R² = nan" not in title, title
    assert "R² undefined" in title, title


def test_vdep_varying_rates_keep_the_numeric_r2(qapp, qtbot):
    """CONTROL — rates that differ across recordings print the number as always."""
    w = VDepWindow(_vdep_session((40, 80)))
    qtbot.addWidget(w)
    w.set_fit("linear")
    good = next(ln for ln in w.details_text().splitlines()
                if ln.startswith("Goodness of fit"))
    assert "R² = " in good and "undefined" not in good and "nan" not in good, good
