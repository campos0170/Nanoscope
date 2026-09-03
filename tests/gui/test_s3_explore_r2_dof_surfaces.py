"""S3 EXPL3-02 — the two surfaces that print an R² both say when it is free.

The rule itself lives in :func:`nanoscope.analysis.stats.r2_dof_tag` /
:func:`~nanoscope.analysis.stats.r2_dof_caveat` (see
``tests/analysis/test_s3_explore_r2_dof.py``). These tests pin that both call sites are
wired to it -- the Plots window's scatter best fit, where the defect was found, and the
voltage-dependence panel's Goodness line, which had the identical hole for the identical
reason and was brought under the rule with it.

The short form rides WITH the number (a title and a legend do not carry the sentence),
the sentence goes in the details pane, and a fit with residual degrees of freedom is
untouched -- an R² over 986 events must not grow a caveat that belongs to an R² over two.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

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


# ================================================================================================
# the Plots scatter
# ================================================================================================

def _scatter_window(qtbot, n: int, *, fit: str = "linear") -> PlotsWindow:
    rng = np.random.default_rng(4)
    x = np.linspace(1.0, 5.0, n)
    df = pd.DataFrame({
        "dwell_s": x * 1e-3,
        "depth_pA": 2.0 * x + 1.0 + (rng.normal(0.0, 0.4, n) if n > 2 else 0.0),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run": df})
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_fit(fit)
    return w


@pytest.mark.parametrize("fit", ["linear", "power law (log-log)"])
def test_two_points_carry_the_qualifier_with_the_number(qapp, qtbot, fit):
    w = _scatter_window(qtbot, 2, fit=fit)
    text = w.details_text()
    line = next(ln for ln in text.splitlines() if "R² = " in ln)
    assert "(exact by construction)" in line, line
    assert "(n = 2)" in line, line


@pytest.mark.parametrize("fit", ["linear", "power law (log-log)"])
def test_two_points_get_the_full_sentence_in_details(qapp, qtbot, fit):
    text = _scatter_window(qtbot, 2, fit=fit).details_text()
    assert "EXACT BY CONSTRUCTION" in text, text
    assert ("log-log line" if fit.startswith("power") else "straight line") in text, text


@pytest.mark.parametrize("fit", ["linear", "power law (log-log)"])
def test_the_qualifier_also_reaches_the_title(qapp, qtbot, fit):
    """The title is what ``Save image`` keeps; the details pane is not exported."""
    w = _scatter_window(qtbot, 2, fit=fit)
    assert "exact by construction" in str(w._plot.titleLabel.text or "")


def test_a_real_sample_is_untouched(qapp, qtbot):
    w = _scatter_window(qtbot, 40)
    text = w.details_text()
    assert "R² = " in text
    assert "exact by construction" not in text, text
    assert "EXACT BY CONSTRUCTION" not in text, text


def test_the_flat_response_sentence_still_stands(qapp, qtbot):
    """The sweep-2 fix on the same three lines must not have been displaced."""
    n = 30
    df = pd.DataFrame({
        "dwell_s": np.linspace(1e-3, 5e-3, n),
        "depth_pA": np.full(n, 42.0),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run": df})
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_fit("linear")
    text = w.details_text()
    assert "R² undefined (y does not vary)" in text, text
    assert "R² = nan" not in text


# ================================================================================================
# the voltage-dependence panel -- the OTHER call site, brought under the same rule
# ================================================================================================

FS = 50_000.0


def _vdoc(tmp_path, *, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _vevents(n: int, *, depth_pA: float) -> pd.DataFrame:
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": "rec", "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": 1.2e-3,
            "baseline_pA": -1000.0, "depth_pA": depth_pA, "frac_depth": 0.1,
            "ecd_pA_s": 0.05, "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _vsession(tmp_path, mvs) -> Session:
    session = Session()
    for mV in mvs:
        doc_id = session.add_document(
            _vdoc(tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
        session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={}, calibration_fingerprint=(1.0, 0.0),
            payload=EventSet(_vevents(30, depth_pA=-100.0 * (mV / 1000.0)),
                             {"analyzed_duration_s": 1.0, "recording": "rec"})))
    return session


def test_a_trend_over_two_recordings_says_it_is_free(qapp, qtbot, tmp_path):
    w = VDepWindow(_vsession(tmp_path, (-200.0, -600.0)))
    qtbot.addWidget(w)
    w.set_metric("median_depth")
    w.set_fit("linear")
    text = w.details_text()
    assert "Goodness of fit:" in text, text
    assert "EXACT BY CONSTRUCTION" in text, text
    assert "a trend has 2 free parameters" in text, text
    assert "2 recordings were fitted" in text, text
    assert "exact by construction" in str(w._plot.titleLabel.text or "")


def test_a_trend_over_four_recordings_is_untouched(qapp, qtbot, tmp_path):
    w = VDepWindow(_vsession(tmp_path, (-200.0, -500.0, -700.0, -1000.0)))
    qtbot.addWidget(w)
    w.set_metric("median_depth")
    w.set_fit("linear")
    text = w.details_text()
    assert "Goodness of fit:  R² = " in text, text
    assert "EXACT BY CONSTRUCTION" not in text, text
    assert "exact by construction" not in str(w._plot.titleLabel.text or "")
