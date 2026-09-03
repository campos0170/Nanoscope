"""GUI tests for the power-spectrum / noise window (pytest-qt, offscreen).

The standalone :class:`~nanoscope.gui.psd_panel.PSDWindow`, exercised the way the
user would for pore quality control on a raw calibrated trace:

* fed a Session with one synthetic current recording (white noise of known
  variance on a DC pedestal), the window computes a Welch PSD and draws a
  non-empty spectrum curve, and its total-RMS read-out recovers the noise sigma;
* changing the Welch segment length re-grids the spectrum and changing the time
  segment re-averages it (fewer segments over a shorter window);
* the user RMS band re-integrates the stored PSD and updates the live read-out;
* the ASD vs PSD toggle changes the plotted y-data to sqrt(PSD);
* *Save image* writes a real PNG, and an empty source shows a message;
* through the real :class:`MainWindow` the ``Analysis > Power spectrum...`` action
  gates on a recording with a current channel + known rate (NOT on a detection)
  and opens a single, lazily-built window that follows the active selection.

Nothing here needs a detection or touches the frozen EventSet schema — PSD is
raw-signal analysis.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.psd_panel import PSDWindow

FS = 50_000.0
N_SAMPLES = 200_000          # 4 s
SIGMA = 3.0
PEDESTAL = 100.0


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(tmp_path, *, name: str = "rec.dat", fs: float | None = FS,
                 n: int = N_SAMPLES, sigma: float = SIGMA,
                 kind: ChannelKind = ChannelKind.CURRENT, units: str = "pA") -> TraceDocument:
    """A recording whose one channel is white noise of std ``sigma`` on a DC pedestal."""
    rng = np.random.default_rng(0)
    sig = (rng.normal(0.0, sigma, n) + PEDESTAL).astype(np.float32)
    info = ChannelInfo(index=0, name=kind.value, raw_units=units, kind=kind)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _fed_window(qtbot, tmp_path, **kw) -> tuple[PSDWindow, str]:
    session = Session()
    doc = _current_doc(tmp_path, **kw)
    doc_id = session.add_document(doc)
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    return w, doc_id


@pytest.fixture
def window(qapp, qtbot, tmp_path):
    w, _ = _fed_window(qtbot, tmp_path)
    return w


# ================================================================================================
# compute + plot
# ================================================================================================

def test_computes_and_plots_a_psd(window):
    w = window
    assert w.has_data()
    freqs, psd = w.psd_result()
    assert freqs.size > 1 and psd.size == freqs.size
    assert freqs[0] == 0.0                                # a one-sided spectrum keeps its DC bin
    assert np.all(psd >= 0.0)

    # the curve is drawn (DC dropped) and non-empty
    assert isinstance(w.curve_item(), pg.PlotDataItem)
    x, y = w.plotted_data()
    assert x.size > 1 and np.all(x > 0.0)
    # the fitted 1/f line is overlaid
    assert isinstance(w._fit_item, pg.PlotDataItem)
    # default log-log axes
    assert w._plot.getViewBox().state["logMode"] == [True, True]


def test_total_rms_recovers_the_noise_sigma(window):
    """The full-band read-out (default RMS band = [0, Nyquist]) ~= the white-noise sigma."""
    w = window
    assert w.summary()["total_rms"] == pytest.approx(SIGMA, rel=0.05)
    assert w.rms_value() == pytest.approx(SIGMA, rel=0.05)     # default band is the whole band
    assert w.units() == "pA"


def test_details_pane_reports_the_noise_regimes(window):
    txt = window.details_text()
    assert "Total RMS noise" in txt
    assert "White floor" in txt
    assert "1/f exponent" in txt
    assert "Welch: nperseg" in txt
    assert "Nyquist" in txt


# ================================================================================================
# controls re-render: segment length + time segment
# ================================================================================================

def test_nperseg_change_regrids_the_spectrum(window):
    w = window
    w.set_nperseg(4096)
    freqs, _ = w.psd_result()
    assert freqs.size == 4096 // 2 + 1                    # FFT grid follows the segment length
    assert w.segment_info()[0] == 4096

    w.set_nperseg(8192)
    freqs2, _ = w.psd_result()
    assert freqs2.size == 8192 // 2 + 1
    assert w.segment_info()[0] == 8192


def test_segment_change_reaverages(window):
    w = window
    w.set_nperseg(4096)                                  # fix the grid so only #segments moves
    nseg_full = w.segment_info()[2]
    x_full, y_full = w.plotted_data()

    w.set_segment(0.0, 1.0)                              # a quarter of the 4 s trace
    nseg_quarter = w.segment_info()[2]
    assert nseg_quarter < nseg_full                      # fewer averaged segments over less data
    # the spectrum genuinely re-rendered (a different window -> different estimate)
    _, y_quarter = w.plotted_data()
    assert not np.allclose(y_full, y_quarter)


def test_use_full_trace_button_resets_the_segment(window):
    w = window
    w.set_segment(0.5, 1.5)
    w._use_full_trace()
    assert w._start_spin.value() == pytest.approx(0.0)
    assert w._end_spin.value() == pytest.approx(N_SAMPLES / FS)


# ================================================================================================
# RMS band read-out
# ================================================================================================

def test_rms_band_readout_updates(window):
    w = window
    full = w.rms_value()
    assert full == pytest.approx(SIGMA, rel=0.05)

    w.set_rms_band(0.0, 1_000.0)                         # a sub-band integrates to less
    sub = w.rms_value()
    assert 0.0 < sub < full
    assert "1000" in w.rms_readout_text()
    assert "pA" in w.rms_readout_text()

    # a band above Nyquist reads zero, never NaN
    w.set_rms_band(30_000.0, 40_000.0)
    assert w.rms_value() == 0.0


# ================================================================================================
# ASD vs PSD toggle
# ================================================================================================

def test_asd_toggle_changes_the_ydata_to_sqrt(window):
    w = window
    assert not w.is_asd()
    _, y_psd = w.plotted_data()

    w.set_asd(True)
    assert w.is_asd()
    x_asd, y_asd = w.plotted_data()
    assert not np.allclose(y_psd, y_asd)                 # a different quantity is drawn
    np.testing.assert_allclose(y_asd, np.sqrt(y_psd), rtol=1e-6)
    assert "√Hz" in w._plot.getAxis("left").labelText     # ASD units label

    w.set_asd(False)
    _, y_back = w.plotted_data()
    np.testing.assert_allclose(y_back, y_psd, rtol=1e-9)


# ================================================================================================
# save image + empty state
# ================================================================================================

def test_save_image_writes_png(window, tmp_path):
    out = window.save_image(str(tmp_path / "psd.png"))
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_empty_source_shows_message_state(qapp, qtbot):
    w = PSDWindow(Session())                              # no recordings at all
    qtbot.addWidget(w)
    assert not w.has_data()
    assert w.psd_result() is None
    assert w.plotted_data() is None
    assert w._message_item is not None
    assert w.rms_readout_text() == "—"


def test_recording_without_a_rate_is_not_offered(qapp, qtbot, tmp_path):
    """A current channel with no known sample rate can't yield a spectrum -> empty state."""
    session = Session()
    doc = _current_doc(tmp_path, fs=None)                # rate-unknown recording
    session.add_document(doc)
    w = PSDWindow(session)
    qtbot.addWidget(w)
    assert w._rec_combo.count() == 0                     # not listed (needs a time base)
    assert not w.has_data()


# ================================================================================================
# wiring through MainWindow
# ================================================================================================

def _install(win: MainWindow, doc: TraceDocument) -> str:
    session = win._session
    doc_id = session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


def test_psd_action_gated_and_opens_single_window(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)

    win._sync_psd_action()
    assert not win._act_psd.isEnabled()                  # no recording -> dead

    doc = _current_doc(tmp_path, name="a.dat")
    doc_id = _install(win, doc)
    win._on_selection(doc_id, 0)
    win._sync_psd_action()
    assert win._act_psd.isEnabled()                      # a current channel + rate -> live

    win._open_psd()
    first = win._psd_window
    assert first is not None
    assert first.has_data()
    assert first.summary()["total_rms"] == pytest.approx(SIGMA, rel=0.05)
    # re-opening focuses the SAME instance (never a second window)
    win._open_psd()
    assert win._psd_window is first


def test_psd_action_dead_without_a_current_channel(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _current_doc(tmp_path, name="v.dat", kind=ChannelKind.VOLTAGE, units="mV")
    doc_id = _install(win, doc)
    win._on_selection(doc_id, 0)
    win._sync_psd_action()
    assert not win._act_psd.isEnabled()                  # voltage-only -> nothing to spectrum


def test_psd_action_dead_without_a_known_rate(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _current_doc(tmp_path, name="norate.dat", fs=None)
    doc_id = _install(win, doc)
    win._on_selection(doc_id, 0)
    win._sync_psd_action()
    assert not win._act_psd.isEnabled()                  # a spectrum needs a time base


def test_psd_window_follows_the_active_selection(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    doc_a = _current_doc(tmp_path, name="a.dat")
    doc_b = _current_doc(tmp_path, name="b.dat")
    id_a = _install(win, doc_a)
    id_b = _install(win, doc_b)

    win._on_selection(id_a, 0)
    win._open_psd()
    assert win._psd_window._current_target() == (id_a, 0)

    win._on_selection(id_b, 0)                            # switch recordings
    assert win._psd_window._current_target() == (id_b, 0)
