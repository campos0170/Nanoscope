"""S4 voltage-series — the two numbers Analysis ▸ I-V / Conductance… prints.

VS-01 — the window printed ``R² = 1.0000`` for a TWO-recording fit with no
degrees-of-freedom qualifier, in the plot title that Save image rasterises. A
straight line has two free parameters, so two points lie on it exactly and R² = 1 is
arithmetic: any two numbers whatsoever, noise included, score the same. The app ships
ONE writer for that qualifier (:func:`nanoscope.analysis.stats.r2_dof_tag` /
:func:`~nanoscope.analysis.stats.r2_dof_caveat`) and the voltage-dependence window and
the Plots scatter already call it; this window was the third fitting surface and
called neither, so one session gave two windows two verdicts on the same n = 2 fit.

VS-07 — for ONE pore the app shows two conductances at once: the Calibration panel's
``G = I₀ / V`` box (a chord through the origin, per recording) and this window's
fitted SLOPE (one number for the series, with a leak intercept). On the owner's three
apoferritin recordings they are 1.71x apart. Neither headline said WHICH conductance
it was and neither surface named the other, so a reader took whichever they saw first.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import iv, stats
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.main_window import MainWindow

WHERE = "(S4 voltage-series: the I-V window qualifies its R² and names the other G)"


def _doc(tmp_path, *, name: str, mV: float, i_pA: float, n: int = 4096) -> TraceDocument:
    """A recording with a measured command channel in declared mV."""
    cur = np.ascontiguousarray(np.full(n, i_pA, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(n, mV, dtype=np.float32))
    infos = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE),
    )
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta, channels=(cur, volt))


def _session(tmp_path, rungs) -> Session:
    session = Session()
    for mV, i_pA in rungs:
        doc = _doc(tmp_path, name=f"pore_{int(abs(mV))}.dat", mV=mV, i_pA=i_pA)
        doc_id = session.add_document(doc)
        # The loader's default calibration for a voltage channel may scale it; pin the
        # units and an identity gain so the fixture's numbers are the fixture's.
        session.set_calibration(doc_id, 1, ChannelCalibration(target_units="mV"),
                                write_sidecar=False)
    return session


def _title(w: IVWindow) -> str:
    label = w.plot_item().titleLabel
    return "" if label is None else str(label.text)


# ================================================================================================
# 1. VS-01 -- the two-recording R²
# ================================================================================================

def test_a_two_recording_r2_carries_the_dof_tag_in_the_exported_title(qapp, qtbot, tmp_path):
    """THE VS-01 REGRESSION. The title is what the PNG keeps, so the tag rides there."""
    w = IVWindow(_session(tmp_path, [(-500.0, -28.0), (-700.0, -49.0)]))
    qtbot.addWidget(w)
    fit = w.fit()
    assert fit is not None and fit.n == 2 and fit.r2 == pytest.approx(1.0), WHERE

    tag = stats.r2_dof_tag(2, 2)
    assert tag == "exact by construction", WHERE
    assert tag in _title(w), f"{_title(w)} {WHERE}"
    # ...and the whole sentence is in the pane, where there is room for it.
    assert "EXACT BY CONSTRUCTION" in w.details_text(), f"{w.details_text()} {WHERE}"
    assert "n = 2 recording(s)" in w.details_text(), WHERE
    # ONE WRITER: the sentence is stats', not a second copy kept here.
    assert stats.r2_dof_caveat(2, 2, model="line", noun="recording") in w.details_text(), WHERE


def test_a_three_recording_fit_is_not_annotated(qapp, qtbot, tmp_path):
    """A guard on a degenerate case, not a running commentary on small samples."""
    w = IVWindow(_session(tmp_path, [(-500.0, -28.0), (-600.0, -35.0), (-700.0, -49.0)]))
    qtbot.addWidget(w)
    assert w.fit().n == 3, WHERE
    assert "exact by construction" not in _title(w), f"{_title(w)} {WHERE}"
    assert "EXACT BY CONSTRUCTION" not in w.details_text(), WHERE
    assert "R² = " in _title(w), WHERE


def test_the_tag_survives_the_png_export(qapp, qtbot, tmp_path):
    w = IVWindow(_session(tmp_path, [(-500.0, -28.0), (-700.0, -49.0)]))
    qtbot.addWidget(w)
    out = tmp_path / "iv.png"
    assert w.save_image(str(out)) is not None, WHERE
    assert out.is_file() and out.stat().st_size > 0, WHERE
    assert "exact by construction" in _title(w), WHERE


def test_both_fitting_windows_say_the_same_thing_about_the_same_n(qapp, qtbot, tmp_path):
    """The defect was two windows, one session, one n = 2, two verdicts."""
    from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
    from nanoscope.core.analysis import AnalysisResult
    from nanoscope.gui.voltage_dep_panel import VDepWindow

    session = _session(tmp_path, [(-500.0, -28.0), (-700.0, -49.0)])
    for k, doc in enumerate(session.documents()):
        df = empty_df()
        for i in range(8):
            df.loc[i] = {
                "event_id": i, "recording": "rec", "pore_id": "P",
                "start_idx": 10 * i, "end_idx": 10 * i + 5, "t_start_s": 1e-3 * i,
                # A dwell that MOVES between the two recordings: a metric identical
                # across the pool has no R2 at all, which is a different sentence.
                "dwell_s": 2e-3 + 1e-3 * k, "baseline_pA": -50.0, "depth_pA": 20.0,
                "frac_depth": 0.4, "ecd_pA_s": 0.05, "max_dev_pA": 30.0,
                "mean_dev_pA": 20.0, "rise_s": 1e-5, "fall_s": 1e-5, "snr": 20.0,
                "n_levels": 1, "filter_limited": False, "accepted": True, "flag": "",
            }
        df = df.astype({c: d for c, (d, _, _) in SCHEMA.items()})
        session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc.doc_id,
            channel=0, params={}, calibration_fingerprint=(1.0, 0.0),
            payload=EventSet(df, {"analyzed_duration_s": 4.0, "recording": "rec"})))

    ivw = IVWindow(session)
    qtbot.addWidget(ivw)
    vdw = VDepWindow(session)
    qtbot.addWidget(vdw)
    vdw.set_metric("median_dwell_s")
    vdw.set_fit("linear")
    vd_title = str(vdw.plot_item().titleLabel.text)
    assert "exact by construction" in vd_title, f"{vd_title} {WHERE}"
    assert "exact by construction" in _title(ivw), f"{_title(ivw)} {WHERE}"


# ================================================================================================
# 2. VS-07 -- two conductances for one pore
# ================================================================================================

#: The owner's three recordings, as the repro measured them: per-bias G of
#: 0.056163 / 0.057992 / 0.070201 nS against a fitted slope of 0.10529 nS, 1.71x
#: apart, driven by a +25.833 pA leak intercept.
_APO = [(-499.9, -28.076), (-599.9, -34.79), (-699.9, -49.133)]


def test_the_headline_says_which_conductance_it_is(qapp, qtbot, tmp_path):
    """THE VS-07 REGRESSION. "G = 0.1053 nS" alone is the ambiguous half."""
    w = IVWindow(_session(tmp_path, _APO))
    qtbot.addWidget(w)
    assert w.fit() is not None, WHERE
    assert "slope" in w._headline.text().lower(), f"{w._headline.text()} {WHERE}"
    assert "slope" in _title(w).lower(), f"{_title(w)} {WHERE}"
    assert "nS" in w._headline.text() and "nS" in _title(w), WHERE


def test_the_details_pane_reconciles_the_two_conductances(qapp, qtbot, tmp_path):
    """The per-bias numbers the Calibration panel shows, their spread, and the leak."""
    session = _session(tmp_path, _APO)
    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()
    assert "G = I₀ / V" in text, f"the other conductance is unnamed {WHERE}"

    # The per-bias values are the Calibration panel's own arithmetic, on the same
    # recordings -- computed here through the module both surfaces use.
    for p in iv.iv_points(session):
        g = iv.conductance_in_nS(p.i_open, p.current_units, p.v_volts * 1000.0)
        assert f"{g:.5g}" in text, f"{g} missing from the reconciliation {WHERE}"

    assert "spread" in text, WHERE
    fit = w.fit()
    assert f"{fit.intercept:.5g}" in text, WHERE           # the leak that separates them
    assert "passes through the origin" in text, WHERE
    assert f"{fit.conductance_nS / np.mean([iv.conductance_in_nS(p.i_open, p.current_units, p.v_volts * 1000.0) for p in iv.iv_points(session)]):.2f}x" in text, WHERE


def test_the_reconciliation_survives_undeclared_current_units(qapp, qtbot, tmp_path):
    """No nS is available -- and the block says so rather than printing a bare number."""
    session = Session()
    for mV, i_pA in ((-500.0, -28.0), (-700.0, -49.0)):
        doc = _doc(tmp_path, name=f"u_{int(abs(mV))}.h5", mV=mV, i_pA=i_pA)
        doc_id = session.add_document(doc)
        session.set_calibration(doc_id, 0, ChannelCalibration(target_units=" "),
                                write_sidecar=False)
        session.set_calibration(doc_id, 1, ChannelCalibration(target_units="mV"),
                                write_sidecar=False)
    w = IVWindow(session)
    qtbot.addWidget(w)
    text = w.details_text()
    assert w.fit() is not None and w.fit().conductance_nS is None, WHERE
    assert "G = I₀ / V" in text, WHERE
    assert "not in nS" in text, f"{text} {WHERE}"


def test_the_window_reached_by_its_own_menu_action_carries_both(qapp, qtbot, tmp_path):
    """Through the REAL Analysis ▸ I-V / Conductance… QAction on a real MainWindow."""
    win = MainWindow()
    qtbot.addWidget(win)
    for mV, i_pA in _APO[:2]:
        doc = _doc(tmp_path, name=f"pore_{int(abs(mV))}.dat", mV=mV, i_pA=i_pA)
        doc_id = win._session.add_document(doc)
        win._create_default_layers(doc)
        win._session.set_calibration(doc_id, 1, ChannelCalibration(target_units="mV"),
                                     write_sidecar=False)
    win._act_iv.trigger()
    w = win._iv_window
    assert w is not None and w.fit() is not None and w.fit().n == 2, WHERE
    assert "exact by construction" in _title(w), f"{_title(w)} {WHERE}"
    assert "slope" in w._headline.text().lower(), f"{w._headline.text()} {WHERE}"
    assert "G = I₀ / V" in w.details_text(), WHERE
