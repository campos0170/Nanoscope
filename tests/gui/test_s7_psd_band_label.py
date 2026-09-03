"""Sweep-7 S7-PSD1: the PSD window's "RMS in band" label restates the integrated band
EXACTLY, at every frequency, and never as a zero-width band beside a non-zero RMS.

Measured on the pre-fix tree (docs/sweep7/repros/S7-PSD1_band_label.py): the edges
came from :func:`nanoscope.analysis.psd.band_support` — the same clip the integral
uses (NE-03) — but were then printed at four SIGNIFICANT digits, so above 9,999 Hz
the label named a different band than the number came from: 12345–12355 Hz read
``(12340–12360 Hz)`` and 24999–25000 Hz read ``(25000–25000 Hz)`` beside 0.007632 pA —
a zero-width band the same window labels "empty band, nothing integrated" when it is
genuinely empty. The 10 kHz-and-above region is where the dielectric rise is read for
pore QC, so it is exactly where a five-digit edge gets typed.

The fix (gui/psd_panel.py): ``_fmt_band_edge`` is the shortest exact spelling, and
``_fmt_band_edges`` prints both edges at the RMS spinboxes' own 0.1 Hz resolution
(the sweep-7 fix-up F-SK-4 replaced the first fix's "fewest decimals within half a
PSD bin", a bin-dependent tolerance that still read a typed 12345.3 Hz back as 12345
on a 0.76 Hz-bin recording) AND keeps the two distinct — a float-noisy Nyquist
(24999.999999999996) still reads 25000, a 0–0.5 Hz band still reads 0–0.5.
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
from nanoscope.gui.psd_panel import PSDWindow, _fmt_band_edge

FS = 50_000.0
_BAND = re.compile(r"\(([0-9.]+)–([0-9.]+) Hz\)")


def _doc(tmp_path, fs: float = FS, n: int = 400_000, sigma: float = 1.49) -> TraceDocument:
    rng = np.random.default_rng(11)
    sig = rng.normal(0.0, sigma, n).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "noise.dat", format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture
def window(qapp, qtbot, tmp_path) -> PSDWindow:
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    return w


def test_edges_above_ten_kilohertz_are_restated_exactly_beside_the_integral(window):
    """The dossier's two bands, plus the formatter alone: no re-rounding anywhere."""
    freqs, _ = window.psd_result()
    for lo, hi in ((12345.0, 12355.0), (24999.0, 25000.0), (10001.0, 10002.0),
                   (12345.0, 20000.0)):
        window.set_rms_band(lo, hi)
        text = window.rms_readout_text()
        sup = psd_mod.band_support(freqs, lo, hi)
        assert sup is not None
        assert f"({lo:g}–{hi:g} Hz)" in text, text
    # A NON-EMPTY band at the top of the axis: a measured figure beside the band it
    # was measured over, never beside a band of zero width.
    window.set_rms_band(24990.0, 25000.0)
    assert window.rms_value() > 0.0
    assert "(24990–25000 Hz)" in window.rms_readout_text()
    window.set_rms_band(24999.0, 25000.0)
    assert "(24999–25000 Hz)" in window.rms_readout_text()
    assert "25000–25000" not in window.rms_readout_text()
    # ...while a GENUINELY empty band keeps its existing words (the spins clamp to
    # Nyquist, so this is the degenerate band band_support names None).
    window.set_rms_band(25000.0, 25000.0)
    assert "(25000–25000 Hz — empty band, nothing integrated)" in window.rms_readout_text()

    assert _fmt_band_edge(12345) == "12345"
    assert _fmt_band_edge(12345.6) == "12345.6"
    assert _fmt_band_edge(24999) == "24999"
    assert _fmt_band_edge(123456) == "123456"
    assert _fmt_band_edge(0.5) == "0.5"
    assert _fmt_band_edge(25000.0) == "25000"          # plain notation, never 2.5e+04


def test_the_label_never_names_a_zero_width_band_beside_a_non_zero_rms(window):
    """Sweep the top decade of the axis with one-decimal spinbox values: whenever the
    integral is non-zero the two printed edges differ, and each is within half a bin
    of the edge band_support handed the integral."""
    freqs, _ = window.psd_result()
    half_bin = 0.5 * float(freqs[1] - freqs[0])
    for lo in np.arange(10000.0, 25000.0, 1234.5):
        for width in (0.5, 1.0, 10.0, 123.4):
            hi = min(lo + width, 25000.0)
            window.set_rms_band(float(lo), float(hi))
            text = window.rms_readout_text()
            sup = psd_mod.band_support(freqs, float(lo), float(hi))
            if sup is None or window.rms_value() == 0.0:
                continue
            m = _BAND.search(text)
            assert m, text
            a, b = float(m.group(1)), float(m.group(2))
            assert a != b, text
            assert abs(a - sup[0]) <= half_bin and abs(b - sup[1]) <= half_bin, (text, sup)


def test_a_float_noisy_nyquist_reads_as_the_round_number_the_spinbox_shows():
    """rfftfreq's Nyquist can be 24999.999999999996; the label must not spell that
    out, nor round the other edge together with it."""
    from nanoscope.gui.psd_panel import _fmt_band_edges   # new with the fix: absent before it
    assert _fmt_band_edges(24999.0, 24999.999999999996) == ("24999", "25000")
    assert _fmt_band_edges(0.0, 0.5) == ("0", "0.5")             # NE-03 stays
    assert _fmt_band_edges(9.5, 10.5) == ("9.5", "10.5")
    assert _fmt_band_edges(24999.1, 24999.3) == ("24999.1", "24999.3")
    # Integer edges: exact spelling, still distinct.
    assert _fmt_band_edges(12345.0, 12355.0) == ("12345", "12355")
