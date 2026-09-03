"""Sweep-7 fix-up F-SK-4: the PSD window's "RMS in band" label restates a typed edge at
the resolution it was typed at — 0.1 Hz, the spinboxes' own — on EVERY recording.

The S7-PSD1 fix printed both edges "at the fewest decimals within half a PSD bin of
the integrated edge", which made the label's precision depend on the recording's bin
width. Measured on the pre-fix-up tree: on the owner's -1000 mV excerpt (0.76 Hz bins,
half = 0.38) a typed 12345.3–12355.7 Hz read ``(12345–12356 Hz)`` and 1000.3–2000.0 read
``(1000–2000 Hz)``; on a 20 000-sample trace (24 Hz bins) 12345.0–12345.5 read
``(12345–12346 Hz)`` and 10000.4–10012.6 read ``(10000–10013 Hz)`` — while
:func:`nanoscope.analysis.psd.band_support` had handed the integral the typed values
unchanged. That is the S7-PSD1 principle (the label beside the integral may not
re-round the edges independently of band_support) surviving at sub-bin magnitude.

The fix (gui/psd_panel.py): ONE constant, ``_BAND_SPIN_DECIMALS``, sets the spinboxes'
``setDecimals`` AND the rounding ``_fmt_band_edges`` applies, so the label cannot be
coarser than what the user can type; the distinct-edges guard stays (a band narrower
than the resolution grows decimals until the two edges separate). The bin width no
longer enters the label at all.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import psd as psd_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import PSDWindow, _BAND_SPIN_DECIMALS, _fmt_band_edges

FS = 50_000.0
_BAND = re.compile(r"\(([0-9.]+)–([0-9.]+) Hz\)")


def _doc(tmp_path, n: int, sigma: float = 1.49) -> TraceDocument:
    rng = np.random.default_rng(3)
    sig = rng.normal(0.0, sigma, n).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / f"noise_{n}.dat", format_name="synthetic",
                         n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _window(qtbot, tmp_path, n: int, nperseg: int | str = "Auto") -> PSDWindow:
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, n))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    w.set_nperseg(nperseg)
    return w


def _typed_edges_print_back_as_typed(w: PSDWindow, bands) -> None:
    freqs, _ = w.psd_result()
    for lo, hi in bands:
        w.set_rms_band(lo, hi)
        sup = psd_mod.band_support(freqs, lo, hi)
        assert sup == (lo, hi), "precondition: band_support passed the typed band through"
        text = w.rms_readout_text()
        m = _BAND.search(text)
        assert m, text
        assert (float(m.group(1)), float(m.group(2))) == (lo, hi), text


def test_a_typed_tenth_of_a_hertz_edge_prints_back_as_typed_on_coarse_bins(qapp, qtbot, tmp_path):
    """The 24 Hz-bin case: the half-bin tolerance rounded 12345.0–12345.5 to 12345–12346
    and 10000.4–10012.6 to 10000–10013."""
    w = _window(qtbot, tmp_path, n=20_000)
    freqs, _ = w.psd_result()
    assert float(freqs[1] - freqs[0]) > 10.0, "precondition: coarse bins"
    _typed_edges_print_back_as_typed(
        w, [(12345.0, 12345.5), (10000.4, 10012.6), (5000.0, 5000.5), (0.0, 0.5)])
    w.set_rms_band(12345.0, 12345.5)
    assert "(12345–12345.5 Hz)" in w.rms_readout_text()
    w.set_rms_band(10000.4, 10012.6)
    assert "(10000.4–10012.6 Hz)" in w.rms_readout_text()
    # A typed band that spans bins: a live figure beside the band exactly as typed.
    w.set_rms_band(10000.4, 10100.6)
    assert w.rms_value() > 0.0
    assert "(10000.4–10100.6 Hz)" in w.rms_readout_text()


def test_a_typed_tenth_of_a_hertz_edge_prints_back_as_typed_on_sub_hertz_bins(
        qapp, qtbot, tmp_path):
    """The owner's-excerpt case (0.76 Hz bins): x.3 / x.7 edges were rounded to integers."""
    w = _window(qtbot, tmp_path, n=400_000, nperseg=65536)
    freqs, _ = w.psd_result()
    assert 0.5 < float(freqs[1] - freqs[0]) < 1.0, "precondition: sub-hertz bins"
    _typed_edges_print_back_as_typed(
        w, [(12345.3, 12355.7), (1000.3, 2000.0), (9999.7, 10000.3), (12345.6, 12355.4)])
    w.set_rms_band(12345.3, 12355.7)
    assert w.rms_value() > 0.0
    assert "(12345.3–12355.7 Hz)" in w.rms_readout_text()
    w.set_rms_band(1000.3, 2000.0)
    assert w.rms_value() > 0.0
    assert "(1000.3–2000 Hz)" in w.rms_readout_text()
    # A float-noisy Nyquist still reads as the round number the spinbox shows, and a
    # non-empty band at the top of the axis is never printed zero-width.
    w.set_rms_band(24999.0, 25000.0)
    assert "(24999–25000 Hz)" in w.rms_readout_text()
    assert w.rms_value() > 0.0


def test_the_spinboxes_and_the_label_share_one_resolution_constant(qapp, qtbot, tmp_path):
    """The invariant, not the symptom: the label rounds to the decimals the spins accept,
    so it cannot restate a typed value more coarsely than the user entered it."""
    w = _window(qtbot, tmp_path, n=20_000)
    assert w._rms_lo_spin.decimals() == _BAND_SPIN_DECIMALS
    assert w._rms_hi_spin.decimals() == _BAND_SPIN_DECIMALS
    assert _fmt_band_edges(12345.3, 12355.7) == ("12345.3", "12355.7")
    assert _fmt_band_edges(1000.3, 2000.0) == ("1000.3", "2000")
    assert _fmt_band_edges(24999.0, 24999.999999999996) == ("24999", "25000")
    assert _fmt_band_edges(0.0, 0.5) == ("0", "0.5")
    # The distinct-edges guard survives: a band narrower than the spin resolution is
    # still non-empty and must not print as one number twice.
    assert _fmt_band_edges(1.02, 1.04) == ("1.02", "1.04")
    assert _fmt_band_edges(24999.96, 24999.999999999996) == ("24999.96", "25000")
