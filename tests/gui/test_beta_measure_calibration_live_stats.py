"""Regression tests for MEAS-01: a calibration edit under live cursors must re-measure.

The trace curve repaints from ``Session.calibrationChanged`` (trace_view.py wires
``refresh_channel`` to it), so the Live statistics readout beside it must re-measure from
the SAME signal — otherwise the two on-screen surfaces disagree by the full gain factor,
and *Add to list* stores the pre-calibration numbers (the exact-commit path hands a
non-preview ``_last_stats`` back unchanged). These tests drive the real widgets offscreen:
a :class:`MainWindow`, its :class:`MeasurePanel`, the ONE shared :class:`TraceView`, and
the very ``Session.set_calibration`` door the Calibration panel's Apply goes through.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADE = (40_000, 60_000)                 # one clean square: depth exactly 30, dwell 0.2 s


# ================================================================================================
# builders (noise-free, so the amplitudes below are EXACT and a stale readout is unambiguous)
# ================================================================================================

def _square_trace() -> np.ndarray:
    """100 pA open pore with ONE exactly-30 pA-deep square over [40k, 60k). No noise:
    a stale-vs-fresh comparison must not hinge on a tolerance."""
    x = np.full(N, 100.0, dtype=np.float32)
    x[_BLOCKADE[0]:_BLOCKADE[1]] = 70.0
    return x


def _doc(tmp_path, *, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(_square_trace(),))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add + layer + select a document's current channel — the real selection path."""
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _enable_measure(window: MainWindow, doc_id: str, i0: int, i1: int):
    """Turn the Measure tool on and drop the cursors on sample span [i0, i1)."""
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))
    return mp


def _apply_gain(window: MainWindow, doc_id: str, channel: int, *,
                gain: float, units: str = "pA") -> None:
    """The exact door the Calibration panel's Apply uses (calibration_panel.py ->
    session.set_calibration). write_sidecar=False so no file lands next to the data."""
    window._session.set_calibration(
        doc_id, channel,
        ChannelCalibration(gain=gain, offset=0.0, target_units=units,
                           provenance=Provenance.MEASURED),
        write_sidecar=False)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# the live readout follows the calibration on screen
# ================================================================================================

def test_calibration_edit_remeasures_live_readout(window, tmp_path):
    """Gain doubles -> the trace repaints to a 60 pA event AND the readout says 60."""
    doc_id, ch = _install(window, _doc(tmp_path, name="cal.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    assert mp.current_measurement()["amplitude"] == pytest.approx(30.0)

    _apply_gain(window, doc_id, ch, gain=2.0)

    # The trace on screen now shows a 60 pA-deep event (calibrated 200 -> 140 pA)...
    chan = window._session.calibrated(doc_id, ch)
    trace_amp = float(np.median(chan.data[:_BLOCKADE[0]])
                      - np.median(chan.data[_BLOCKADE[0]:_BLOCKADE[1]]))
    assert trace_amp == pytest.approx(60.0)
    # ...and the Live statistics agree with it, in the dict AND in the rendered text.
    stats = mp.current_measurement()
    assert stats is not None
    assert stats["amplitude"] == pytest.approx(60.0), \
        "the readout kept the pre-calibration amplitude — two surfaces disagree"
    assert mp.readout_value("amplitude") == "60 pA"


def test_add_to_list_after_calibration_edit_saves_new_gain_numbers(window, tmp_path):
    """The saved row carries numbers measured under the calibration on screen."""
    doc_id, ch = _install(window, _doc(tmp_path, name="cal.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    _apply_gain(window, doc_id, ch, gain=2.0)

    assert mp.add_current() is True
    saved = float(mp.measurements_df["amplitude"].iloc[-1])
    assert saved == pytest.approx(60.0), \
        f"Add to list stored the pre-calibration amplitude ({saved})"


def test_units_change_updates_readout_value_and_suffix(window, tmp_path):
    """gain 0.001 / target nA: the crosshair, the curve and this readout all say nA now,
    so the readout must show 0.03 nA — value and unit label together."""
    doc_id, ch = _install(window, _doc(tmp_path, name="cal.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)

    _apply_gain(window, doc_id, ch, gain=0.001, units="nA")

    stats = mp.current_measurement()
    assert stats is not None
    assert stats["amplitude"] == pytest.approx(0.03)
    assert mp.readout_value("amplitude") == "0.03 nA"


def test_calibration_change_on_other_document_leaves_readout_alone(window, tmp_path):
    """The handler is targeted: another recording's calibration edit must not clear or
    rewrite the readout describing THIS channel."""
    doc_id, _ch = _install(window, _doc(tmp_path, name="a.dat"))
    other_doc = _doc(tmp_path, name="b.dat")
    other_id = window._session.add_document(other_doc)
    window._create_default_layers(other_doc)
    window._on_selection(doc_id, 0)          # measure on a.dat

    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    assert mp.current_measurement()["amplitude"] == pytest.approx(30.0)

    _apply_gain(window, other_id, 0, gain=5.0)

    stats = mp.current_measurement()
    assert stats is not None, "an unrelated calibration edit blanked the readout"
    assert stats["amplitude"] == pytest.approx(30.0)
