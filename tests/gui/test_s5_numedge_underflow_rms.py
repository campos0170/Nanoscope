"""Sweep-5 numeric-edge regression NE-07, the PSD window's side: no surface prints a
confident zero for a live signal whose spectrum underflowed float64.

The calibrated pipeline is float32-bound (TraceDocument: "float32 is BINDING", and
ChannelCalibration refuses a gain that underflows float32), so today's loaders cannot
hand the window a subnormal float64 trace — welch_psd's own RuntimeWarning (tested in
tests/analysis/test_s5_numedge_psd_contracts.py) is the fix the API repro exercises.
These tests pin the window's HONESTY WIRING for the same state, exactly as
tests/gui/test_nonfinite_gui.py does for the NaN case: feed the real widget a curve in
the underflow state and read every surface that used to say "0 pA" — the Details
pane, the RMS read-out, the legend text, and the empty-plot explanation. If a float64
data path ever appears, the surfaces are already unable to lie.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import warnings

import numpy as np
import pytest

from nanoscope.analysis import filters as nsfilters
from nanoscope.analysis import psd as psd_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import PSDCurve, PSDWindow

FS = 50_000.0


def _real_window(qtbot, tmp_path) -> PSDWindow:
    """A PSDWindow fed a real recording, so spins/ranges are configured as shipped."""
    rng = np.random.default_rng(0)
    sig = (100.0 + rng.normal(0.0, 3.0, 100_000)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "rec.dat", format_name="synthetic",
                         n_samples=sig.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(sig,)))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    return w


def _underflow_curve(label: str = "sub.dat") -> PSDCurve:
    """A curve in the state _compute_curve flags: live signal, all-zero finite PSD."""
    rng = np.random.default_rng(7)
    seg = 1e-310 + rng.normal(0.0, 1.0, 200_000) * 2e-312
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # welch_psd's own NE-07 warning
        freqs, psd = psd_mod.welch_psd(seg, FS)
    assert np.isfinite(psd).all() and not psd.any()
    # the exact detection expression _compute_curve applies
    constant = bool(np.isfinite(seg).all() and seg.size and seg.min() == seg.max())
    underflow = bool(not constant and np.isfinite(seg).all() and psd.size and not psd.any())
    assert underflow
    time_rms = float(nsfilters.noise_rms(np.asarray(seg, dtype=np.float64)))
    assert time_rms > 0.0
    return PSDCurve(key="k", doc_id="d", channel=0, label=label, colour="#fff",
                    freqs=freqs, psd=psd,
                    summary=psd_mod.noise_summary(freqs, psd, FS), units="pA", fs=FS,
                    constant=constant, underflow=underflow, time_rms=time_rms)


@pytest.fixture
def underflowed(qapp, qtbot, tmp_path):
    w = _real_window(qtbot, tmp_path)
    curve = _underflow_curve()
    w._curves = [curve]
    w._freqs, w._psd, w._summary = curve.freqs, curve.psd, curve.summary
    return w, curve


def test_details_say_below_representable_never_0_pa(underflowed):
    w, curve = underflowed
    text = w._curve_details(curve, header=False)
    assert "Total RMS noise:   below-representable" in text
    assert "0 pA" not in text
    # the honest tiny number travels with the refusal
    assert f"{curve.time_rms:.4g}" in text
    assert "square to below float64" in text
    assert "never a quiet pore" in text


def test_the_rms_readout_refuses_the_confident_zero(underflowed):
    w, _curve = underflowed
    w._update_rms_readout()
    text = w.rms_readout_text()
    assert "below-representable" in text
    assert not text.startswith("0 ")


def test_the_legend_entry_carries_the_refusal_not_0_pa_rms(underflowed):
    w, curve = underflowed
    assert "(RMS below-representable)" in w._curve_legend_text(curve, asd=False)
    assert "0 pA" not in w._curve_legend_text(curve, asd=False)


def test_the_empty_loglog_plot_names_the_representation_limit(underflowed):
    """All-zero bins draw nothing on log axes; the explanation must name underflow,
    not send the user to the log-log toggle."""
    w, curve = underflowed
    reason = w._nothing_drawn_reason()
    assert "below float64" in reason
    assert "not a quiet pore" in reason
    assert f"{curve.time_rms:.4g}" in reason         # the measured time-domain RMS
    assert "log-log toggle" not in reason


def test_a_constant_curve_keeps_its_own_distinct_explanation(qapp, qtbot, tmp_path):
    """constant and underflow are DIFFERENT states with different remedies; the flat
    line's branch must not be absorbed by the new one."""
    w = _real_window(qtbot, tmp_path)
    (curve,) = w.curves()
    assert not curve.underflow and not curve.constant
    flat = np.full(4096, 42.0)
    freqs, psd = psd_mod.welch_psd(flat, FS)
    c = PSDCurve(key="k2", doc_id="d2", channel=0, label="flat.dat", colour="#fff",
                 freqs=freqs, psd=psd, summary=psd_mod.noise_summary(freqs, psd, FS),
                 units="pA", fs=FS, constant=True)
    text = w._curve_details(c, header=False)
    assert "this 0 is arithmetic" in text
    assert "below-representable" not in text
