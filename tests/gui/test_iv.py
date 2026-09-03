"""GUI tests for the I-V / conductance feature (pytest-qt, offscreen).

Two deliverables, both exercised the way the user would:

* the standalone :class:`~nanoscope.gui.iv_panel.IVWindow` — fed a Session with
  three synthetic single-current-channel recordings whose open-pore currents lie
  on a known ohmic line, its fit reports the known CONDUCTANCE (nS), scatters a
  point per recording and overlays the I-V line; empty / single-recording states
  show a message instead of crashing; and the ``Analysis ▸ I-V / Conductance…``
  action gates + opens a single window through the real :class:`MainWindow`.

* the Calibration panel's live conductance read-out — for a selected current
  channel it shows G = I₀/V in nS equal to an independent computation, and it
  recomputes when the current gain is edited (the deferred current-gain
  workflow: dial the gain until G is physical).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.analysis import iv
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.main_window import MainWindow


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(tmp_path, *, i_open_pA: float, nominal_mV: float, name: str,
                 n: int = 4096, noise: float = 0.0) -> TraceDocument:
    """A current-only recording whose open-pore median is ``i_open_pA``."""
    if noise:
        rng = np.random.default_rng(0)
        arr = np.ascontiguousarray(i_open_pA + rng.normal(0.0, noise, n), dtype=np.float32)
    else:
        arr = np.ascontiguousarray(np.full(n, i_open_pA, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=nominal_mV, voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(arr,))


def _ohmic_session(tmp_path, *, G_nS: float = 100.0, leak_pA: float = 0.0) -> Session:
    """3 current-only docs on I[pA] = G[nS]·V[mV] + leak, at -200/-500/-1000 mV."""
    session = Session()
    for mV in (-200.0, -500.0, -1000.0):
        session.add_document(_current_doc(
            tmp_path, i_open_pA=G_nS * mV + leak_pA, nominal_mV=mV, name=f"pore_{int(-mV)}.dat"))
    return session


def _read_nS(text: str) -> float | None:
    """Parse the '<value> nS' read-out label to a float (None on a dash)."""
    text = text.strip()
    if not text or text.startswith("—"):
        return None
    return float(text.split()[0])


# ================================================================================================
# IVWindow — the fit
# ================================================================================================

@pytest.fixture
def ohmic_window(qapp, qtbot, tmp_path):
    session = _ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0)
    w = IVWindow(session)
    qtbot.addWidget(w)
    return w


def test_iv_window_fit_reports_known_conductance(ohmic_window):
    w = ohmic_window
    assert len(w.points()) == 3
    fit = w.fit()
    assert fit is not None
    assert fit.conductance_nS == pytest.approx(100.0, rel=1e-3)
    assert fit.intercept == pytest.approx(-10.0, abs=0.5)   # the leak
    assert fit.r2 == pytest.approx(1.0, abs=1e-6)
    assert fit.n == 3


def test_iv_window_draws_scatter_and_fit_line(ohmic_window):
    w = ohmic_window
    assert isinstance(w._scatter_item, pg.ScatterPlotItem)
    assert w._scatter_item.data.size == 3          # one point per recording
    assert isinstance(w._fit_item, pg.PlotDataItem)
    items = list(w.plot_item().listDataItems())
    assert w._scatter_item in items and w._fit_item in items
    # The conductance is announced on the plot title, the headline and the details.
    assert "nS" in w.plot_item().titleLabel.text
    assert "nS" in w._headline.text()
    assert "Conductance" in w.details_text()


def test_iv_window_refresh_picks_up_new_recording(qapp, qtbot, tmp_path):
    session = Session()
    session.add_document(_current_doc(tmp_path, i_open_pA=-20000.0, nominal_mV=-200.0, name="a.dat"))
    w = IVWindow(session)
    qtbot.addWidget(w)
    assert len(w.points()) == 1
    assert w.fit() is None   # one point cannot fit a line
    assert "another" in w.details_text().lower()

    # Open a second recording at a different voltage and Refresh -> a fit appears.
    session.add_document(_current_doc(tmp_path, i_open_pA=-100000.0, nominal_mV=-1000.0, name="b.dat"))
    w.refresh()
    assert len(w.points()) == 2
    assert w.fit() is not None
    assert w.fit().conductance_nS == pytest.approx(100.0, rel=1e-6)


def test_iv_window_empty_session_shows_message(qapp, qtbot):
    w = IVWindow(Session())
    qtbot.addWidget(w)
    assert w.points() == []
    assert w.fit() is None
    assert w._scatter_item is None
    assert "No usable recordings" in w.details_text()


def test_iv_window_measured_voltage_channel(qapp, qtbot, tmp_path):
    """Two ABF-shaped docs (current + voltage channel): the fit uses the MEASURED voltage."""
    session = Session()
    for v_raw_mV, i_pA in ((-5000.0, -50000.0), (-10000.0, -100000.0)):   # 0.1 gain -> -500/-1000 mV
        cur = np.ascontiguousarray(np.full(4096, i_pA, dtype=np.float32))
        volt = np.ascontiguousarray(np.full(4096, v_raw_mV, dtype=np.float32))
        cur_info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
        volt_info = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
        meta = RecordingMeta(
            path=tmp_path / f"abf_{int(-v_raw_mV)}.abf", format_name="synthetic", n_samples=4096,
            sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
            channels=(cur_info, volt_info), nominal_voltage_mV=round(v_raw_mV * 0.1 / 10) * 10)
        session.add_document(TraceDocument(meta=meta, channels=(cur, volt)))
    w = IVWindow(session)
    qtbot.addWidget(w)
    assert all(p.voltage_source == "measured" for p in w.points())
    # I[pA] = 100 nS * V[mV]: -50000 pA @ -500 mV, -100000 pA @ -1000 mV -> 100 nS.
    assert w.fit().conductance_nS == pytest.approx(100.0, rel=1e-3)


# ================================================================================================
# IVWindow — wiring through MainWindow
# ================================================================================================

def _install(window: MainWindow, doc: TraceDocument) -> str:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    return doc_id


def test_iv_action_gated_and_opens_single_window(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)

    win._sync_iv_action()
    assert not win._act_iv.isEnabled()          # no documents -> dead

    _install(win, _current_doc(tmp_path, i_open_pA=-20000.0, nominal_mV=-200.0, name="a.dat"))
    win._sync_iv_action()
    assert win._act_iv.isEnabled()              # a usable recording -> live

    win._open_iv()
    first = win._iv_window
    assert first is not None
    assert len(first.points()) == 1
    # Re-opening focuses the SAME window instance (never a second one).
    win._open_iv()
    assert win._iv_window is first


def test_iv_action_dead_without_voltage(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    arr = np.ascontiguousarray(np.full(2048, -30.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "no_v.dat", format_name="synthetic", n_samples=2048,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,),
        nominal_voltage_mV=None)
    _install(win, TraceDocument(meta=meta, channels=(arr,)))
    win._sync_iv_action()
    assert not win._act_iv.isEnabled()          # current but no known voltage -> dead


# ================================================================================================
# Calibration panel — live conductance read-out
# ================================================================================================

@pytest.fixture
def cal_panel(qapp, qtbot):
    session = Session()
    panel = CalibrationPanel(session)
    qtbot.addWidget(panel)
    return panel, session


def test_calibration_conductance_equals_I_over_V(cal_panel, tmp_path):
    panel, session = cal_panel
    doc = _current_doc(tmp_path, i_open_pA=-50.0, nominal_mV=-500.0, name="c.dat")
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)

    # Box shown for a current-channel doc (isHidden, not isVisible: the panel itself is never
    # show()n offscreen, so isVisible() is always False; isHidden() tracks the setVisible flag).
    assert not panel._cond_box.isHidden()
    assert _read_nS(panel._cond_voltage.text()) == pytest.approx(-500.0)
    # Identity gain -> I₀ = -50 pA, G = -50/-500 = 0.1 nS (pA/mV -> nS, factor 1).
    g0 = _read_nS(panel._cond_g.text())
    assert g0 == pytest.approx(iv.conductance_in_nS(-50.0, "pA", -500.0))
    assert g0 == pytest.approx(0.1)


def test_calibration_conductance_updates_when_gain_changes(cal_panel, tmp_path):
    panel, session = cal_panel
    doc = _current_doc(tmp_path, i_open_pA=-50.0, nominal_mV=-500.0, name="d.dat")
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)
    g0 = _read_nS(panel._cond_g.text())

    # Dial a 25x current gain: I₀ scales 25x, so does the conductance, live (no Apply needed).
    panel._gain.setText("25")
    g1 = _read_nS(panel._cond_g.text())
    assert g1 == pytest.approx(g0 * 25.0, rel=1e-6)
    assert g1 == pytest.approx(iv.conductance_in_nS(25.0 * -50.0, "pA", -500.0))
    assert _read_nS(panel._cond_iopen.text()) == pytest.approx(-1250.0)


def test_calibration_conductance_uses_measured_voltage_channel(cal_panel, tmp_path):
    panel, session = cal_panel
    cur = np.ascontiguousarray(np.full(4096, -250.0, dtype=np.float32))
    volt = np.ascontiguousarray(np.full(4096, -7000.0, dtype=np.float32))  # 0.1 gain -> -700 mV
    cur_info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    volt_info = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / "e.abf", format_name="synthetic", n_samples=4096,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
        channels=(cur_info, volt_info), nominal_voltage_mV=-700.0)
    doc = TraceDocument(meta=meta, channels=(cur, volt))
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)   # select the current channel

    assert _read_nS(panel._cond_voltage.text()) == pytest.approx(-700.0, abs=1e-2)
    # G = -250 pA / -700 mV = 0.357 nS (compared at the label's .4g display precision).
    assert _read_nS(panel._cond_g.text()) == pytest.approx(
        iv.conductance_in_nS(-250.0, "pA", -700.0), rel=1e-3)


def test_calibration_conductance_hidden_without_current_channel(cal_panel, tmp_path):
    panel, session = cal_panel
    volt = np.ascontiguousarray(np.full(2048, -700.0, dtype=np.float32))
    info = ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / "volt_only.dat", format_name="synthetic", n_samples=2048,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(volt,))
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)
    assert panel._cond_box.isHidden()   # no current channel -> the read-out is hidden


# ================================================================================================
# Medium-8 — TWO OPEN-PORE CURRENTS, AND THE CROSS-REFERENCE HAS TO RUN BOTH WAYS
# ================================================================================================
# The event table's Summary row and its `baseline_pA` column tip were both made to point AT
# this window and say which number to quote where. This window pointed nowhere: its axis
# read "Open-pore current I₀ (nA)" and its per-point read-out "I₀ = … [voltage_source]",
# with nothing to say a different open-pore current exists. Driving the real window over
# the three apoferritin recordings found no label or tooltip mentioning the event table.
#
# The two are computed differently on purpose — one I₀ for the whole recording from
# `analysis.iv`'s estimator, against the median of the per-event stamped `baseline_pA`
# under that detection's `baseline_fingerprint`. On the owner's recordings they differ by
# up to 5.1% (−49.13 vs −51.78 pA at −700 mV).

def test_the_iv_window_says_the_event_table_reports_a_different_open_pore(ohmic_window):
    text = ohmic_window.details_text()
    assert "baseline_pA" in text, text
    assert "5.1%" in text, text
    # ...and which number goes where, since that is the actionable half.
    assert "beside a conductance" in text and "beside a depth" in text, text


def test_the_note_is_on_the_plot_itself_not_only_in_the_details(ohmic_window):
    """A reader who exports the PNG never opens the details pane; a reader who hovers
    the plot is the one about to quote the number off it."""
    tip = ohmic_window.plot_item().getViewBox().toolTip()
    assert "event table" in tip, tip
    assert "NOT this number" in tip, tip


def test_both_directions_quote_the_same_measured_divergence():
    """One fact, two surfaces. If the numbers in the two sentences ever drift apart the
    cross-reference stops being a cross-reference."""
    from nanoscope.gui.iv_panel import _I0_CROSS_REFERENCE
    from nanoscope.gui.panels import StatsPanel

    theirs = StatsPanel._ROW_TIPS["median_baseline_pA"]
    for token in ("5.1%", "−49.13", "−51.78"):
        assert token in _I0_CROSS_REFERENCE, token
        assert token in theirs, token
