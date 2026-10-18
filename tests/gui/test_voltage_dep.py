"""GUI tests for the voltage-dependence feature (pytest-qt, offscreen).

The standalone :class:`~nanoscope.gui.voltage_dep_panel.VDepWindow`, exercised the
way the user would after batch-detecting a voltage series:

* fed a Session with four synthetic recordings at -200/-500/-700/-1000 mV, each
  carrying a stored ``detect.schmitt`` detection whose per-recording metrics were
  injected on a KNOWN trend, the window scatters one point per recording and its
  fit reads that trend back -- the ``A*exp(V/V0)`` capture-rate model and a straight
  line for a metric linear in V;
* the metric and fit combos re-render (a new metric re-gathers the points, a new fit
  overlays a new curve), and the |V| / log-y / accepted-only toggles change the plot;
* empty and single-recording sessions show a message instead of crashing, and
  *Save image* writes a PNG;
* through the real :class:`MainWindow` the ``Analysis > Voltage dependence...`` action
  gates on >= 2 ready recordings and opens a single, lazily-built window.

Metrics are injected DIRECTLY as hand-built :class:`~nanoscope.analysis.events.EventSet`
frames (not produced by a detector), so the plotted/fitted numbers are checked against
exactly what went in.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.voltage_dep_panel import VDepWindow

_EXP_A = 10.0
_EXP_V0 = -0.5
_LADDER_MV = (-200.0, -500.0, -700.0, -1000.0)
_N_PER_DOC = 50


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(tmp_path, *, nominal_mV: float, name: str, n: int = 2048) -> TraceDocument:
    arr = np.ascontiguousarray(np.full(n, -200.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _events_df(dwell_s, depth_pA, frac_depth, accepted, *, recording="rec") -> pd.DataFrame:
    n = len(dwell_s)
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i, "recording": recording, "pore_id": "P",
            "start_idx": 100 * i, "end_idx": 100 * i + 50,
            "t_start_s": 1e-3 * i, "dwell_s": float(dwell_s[i]),
            "baseline_pA": -1000.0, "depth_pA": float(depth_pA[i]),
            "frac_depth": float(frac_depth[i]), "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0, "mean_dev_pA": 100.0,
            "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
            "n_levels": 1, "filter_limited": False,
            "accepted": bool(accepted[i]), "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


def _constant_df(*, n: int, dwell_s: float, depth_pA: float, frac_depth: float) -> pd.DataFrame:
    return _events_df([dwell_s] * n, [depth_pA] * n, [frac_depth] * n, [True] * n)


def _install_detection(session: Session, doc_id: str, df: pd.DataFrame, *, duration_s: float,
                       channel: int = 0, module_id: str = "detect.schmitt") -> None:
    ev = EventSet(df, {"analyzed_duration_s": float(duration_s), "recording": "rec"})
    session.add_result(AnalysisResult(
        module_id=module_id, module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))


def _ladder_session(tmp_path) -> Session:
    """4 docs at the voltage ladder; median_depth = -100*V, capture_rate = A*exp(V/V0)."""
    session = Session()
    for mV in _LADDER_MV:
        v = mV / 1000.0
        doc_id = session.add_document(
            _current_doc(tmp_path, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
        rate = _EXP_A * np.exp(v / _EXP_V0)
        df = _constant_df(
            n=_N_PER_DOC, dwell_s=(v + 1.2) * 1e-3, depth_pA=-100.0 * v, frac_depth=-0.1 * v)
        _install_detection(session, doc_id, df, duration_s=_N_PER_DOC / rate)
    return session


@pytest.fixture
def ladder_window(qapp, qtbot, tmp_path):
    w = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(w)
    return w


# ================================================================================================
# one point per recording
# ================================================================================================

def test_window_plots_one_point_per_recording(ladder_window):
    w = ladder_window
    assert len(w.points()) == len(_LADDER_MV)
    # A PlotDataItem (symbol markers, no line) is used so the log-y toggle transforms the
    # points with the axis; a bare ScatterPlotItem has no setLogMode and would misplot.
    assert isinstance(w._scatter_item, pg.PlotDataItem)
    assert w._scatter_item.yData.size == len(_LADDER_MV)   # one marker per recording
    # the per-recording table lists every gathered point
    assert w._table.rowCount() == len(_LADDER_MV)


def test_table_lists_voltage_in_mV(ladder_window):
    w = ladder_window
    mV_col = {float(w._table.item(r, 1).text()) for r in range(w._table.rowCount())}
    assert mV_col == set(_LADDER_MV)


# ================================================================================================
# fits report the injected trend
# ================================================================================================

def test_exponential_fit_reports_capture_rate_model(ladder_window):
    """Default metric is capture rate; the exponential fit reads back A and V0."""
    w = ladder_window
    w.set_fit("exponential")
    fit = w.fit()
    assert fit is not None
    assert fit["model"] == "exponential"
    assert fit["params"]["A"] == pytest.approx(_EXP_A, rel=1e-6)
    assert fit["params"]["V0"] == pytest.approx(_EXP_V0, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    # the recovered V0 is announced on the headline + plot title + details
    assert "V" in w._headline.text() and "0" in w._headline.text()
    assert "nS" not in w.details_text()
    assert isinstance(w._fit_item, pg.PlotDataItem)      # the curve is overlaid


def test_linear_fit_reports_injected_depth_slope(ladder_window):
    w = ladder_window
    w.set_metric("median_depth")     # median_depth = -100 * V
    w.set_fit("linear")
    fit = w.fit()
    assert fit["model"] == "linear"
    assert fit["params"]["slope"] == pytest.approx(-100.0, rel=1e-6)
    assert fit["params"]["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert "slope" in w._headline.text()


# ================================================================================================
# the metric + fit combos re-render
# ================================================================================================

def test_metric_combo_regathers_points(ladder_window):
    w = ladder_window
    w.set_fit("none")
    w.set_metric("capture_rate_hz")
    cap_vals = sorted(p.value for p in w.points())
    w.set_metric("median_depth")
    depth_vals = sorted(p.value for p in w.points())
    assert cap_vals != depth_vals                       # a new metric re-gathered
    assert depth_vals == pytest.approx(sorted(-100.0 * (mV / 1000.0) for mV in _LADDER_MV))


def test_fit_combo_overlays_and_removes_the_curve(ladder_window):
    w = ladder_window
    w.set_metric("median_depth")
    w.set_fit("none")
    assert w.fit() is None
    assert w._fit_item is None
    w.set_fit("linear")
    assert w.fit() is not None
    assert isinstance(w._fit_item, pg.PlotDataItem)
    w.set_fit("none")
    assert w.fit() is None
    assert w._fit_item is None


# ================================================================================================
# axis + curation toggles
# ================================================================================================

def test_abs_voltage_toggle_flips_x_and_v0_sign(ladder_window):
    w = ladder_window
    w.set_fit("exponential")
    assert w.fit()["params"]["V0"] == pytest.approx(_EXP_V0, rel=1e-6)   # signed V -> -0.5
    w.set_abs_voltage(True)
    xs = w._scatter_item.xData
    assert np.all(xs > 0)                                # plotted against |V|
    assert w.fit()["params"]["V0"] == pytest.approx(-_EXP_V0, rel=1e-6)  # |V| -> +0.5


def test_log_y_toggle_sets_log_axis(ladder_window):
    w = ladder_window
    w.set_log_y(True)
    assert w.plot_item().getAxis("left").logMode is True
    assert len(w.points()) == len(_LADDER_MV)           # all rates positive -> nothing dropped
    w.set_log_y(False)
    assert w.plot_item().getAxis("left").logMode is False


def test_accepted_only_toggle_regathers(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_current_doc(tmp_path, nominal_mV=-500.0, name="mixed.dat"))
    df = _events_df(
        dwell_s=[1e-3, 1e-3, 1e-3, 9e-3, 9e-3, 9e-3],
        depth_pA=[20.0] * 6, frac_depth=[0.02] * 6,
        accepted=[True, True, True, False, False, False])
    _install_detection(session, doc_id, df, duration_s=1.0)
    w = VDepWindow(session)
    qtbot.addWidget(w)

    w.set_metric("median_dwell_s")
    w.set_accepted_only(True)
    assert w.points()[0].value == pytest.approx(1e-3)   # accepted rows only
    w.set_accepted_only(False)
    assert w.points()[0].value == pytest.approx(5e-3)   # median over all six


# ================================================================================================
# empty / single-recording states
# ================================================================================================

def test_empty_session_shows_message(qapp, qtbot):
    w = VDepWindow(Session())
    qtbot.addWidget(w)
    assert w.points() == []
    assert w.fit() is None
    assert w._scatter_item is None
    assert "No usable recordings" in w.details_text()


def test_single_recording_cannot_fit(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_current_doc(tmp_path, nominal_mV=-500.0, name="one.dat"))
    _install_detection(session, doc_id, _constant_df(
        n=5, dwell_s=1e-3, depth_pA=50.0, frac_depth=0.05), duration_s=1.0)
    w = VDepWindow(session)
    qtbot.addWidget(w)
    assert len(w.points()) == 1
    w.set_fit("linear")
    assert w.fit() is None                              # one point is not a trend
    assert ">= 2" in w.details_text() or ">=2" in w.details_text()


def test_refresh_picks_up_a_newly_detected_recording(qapp, qtbot, tmp_path):
    session = Session()
    a = session.add_document(_current_doc(tmp_path, nominal_mV=-200.0, name="a.dat"))
    _install_detection(session, a, _constant_df(
        n=3, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=1.0)
    w = VDepWindow(session)
    qtbot.addWidget(w)
    assert len(w.points()) == 1

    b = session.add_document(_current_doc(tmp_path, nominal_mV=-1000.0, name="b.dat"))
    _install_detection(session, b, _constant_df(
        n=9, dwell_s=2e-4, depth_pA=100.0, frac_depth=0.10), duration_s=1.0)
    w.refresh()
    assert len(w.points()) == 2


# ================================================================================================
# save image
# ================================================================================================

def test_save_image_writes_png(ladder_window, tmp_path):
    out = ladder_window.save_image(str(tmp_path / "vdep.png"))
    assert out is not None
    assert out.exists() and out.stat().st_size > 0


# ================================================================================================
# wiring through MainWindow
# ================================================================================================

def _install(window: MainWindow, doc: TraceDocument) -> str:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    return doc_id


def test_vdep_action_gated_and_opens_single_window(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)

    win._sync_vdep_action()
    assert not win._act_vdep.isEnabled()                # no documents -> dead

    # One ready recording (voltage + detection) is not enough: a single point is no trend.
    a = _install(win, _current_doc(tmp_path, nominal_mV=-200.0, name="a.dat"))
    _install_detection(win._session, a, _constant_df(
        n=4, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=1.0)
    win._sync_vdep_action()
    assert not win._act_vdep.isEnabled()                # only 1 ready -> still dead

    # A second ready recording lights the action.
    b = _install(win, _current_doc(tmp_path, nominal_mV=-1000.0, name="b.dat"))
    _install_detection(win._session, b, _constant_df(
        n=8, dwell_s=2e-4, depth_pA=100.0, frac_depth=0.10), duration_s=1.0)
    win._sync_vdep_action()
    assert win._act_vdep.isEnabled()

    win._open_vdep()
    first = win._vdep_window
    assert first is not None
    assert len(first.points()) == 2
    # re-opening focuses the SAME instance (never a second window)
    win._open_vdep()
    assert win._vdep_window is first


def test_vdep_action_lights_for_an_ml_only_series_and_names_the_detector(qapp, qtbot, tmp_path):
    """MEASURED BEFORE: the gate and the plot read only ``detect.schmitt``, so a project whose
    every table came from the ML detector had the action greyed out."""
    win = MainWindow()
    qtbot.addWidget(win)
    for name, mV in (("a.dat", -500.0), ("b.dat", -700.0)):
        doc_id = _install(win, _current_doc(tmp_path, nominal_mV=mV, name=name))
        _install_detection(win._session, doc_id, _constant_df(
            n=6, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=1.0,
            module_id="detect.keras")
    win._sync_vdep_action()
    assert win._act_vdep.isEnabled()

    win._open_vdep()
    assert len(win._vdep_window.points()) == 2
    assert any("Detector:" in line and "Keras" in line
               for line in win._vdep_window._provenance_lines(win._vdep_window._series))


def test_workspace_capture_records_the_detector_the_series_pooled(qapp, qtbot, tmp_path):
    """The saved point sources named detect.schmitt whatever detector the window pooled, so a
    Keras series was fingerprint-checked against (and mislabelled as) a Schmitt table."""
    from nanoscope.core import analysis_workspace as workspace

    win = MainWindow()
    qtbot.addWidget(win)
    for name, mV in (("a.dat", -500.0), ("b.dat", -700.0), ("c.dat", -900.0)):
        doc_id = _install(win, _current_doc(tmp_path, nominal_mV=mV, name=name))
        _install_detection(win._session, doc_id, _constant_df(
            n=6, dwell_s=1e-3, depth_pA=20.0, frac_depth=0.02), duration_s=1.0,
            module_id="detect.keras")
    win._open_vdep()

    artifact = workspace.capture_voltage_dependence(win._vdep_window)

    sources = artifact.payload["point_sources"]
    assert len(sources) == 3
    assert {s["module_id"] for s in sources} == {"detect.keras"}
    assert all(s["result_fingerprint"] for s in sources)


def test_vdep_action_dead_without_a_detection(qapp, qtbot, tmp_path):
    """Two recordings with a voltage but no detection -> nothing to place, action dead."""
    win = MainWindow()
    qtbot.addWidget(win)
    _install(win, _current_doc(tmp_path, nominal_mV=-200.0, name="a.dat"))
    _install(win, _current_doc(tmp_path, nominal_mV=-500.0, name="b.dat"))
    win._sync_vdep_action()
    assert not win._act_vdep.isEnabled()
