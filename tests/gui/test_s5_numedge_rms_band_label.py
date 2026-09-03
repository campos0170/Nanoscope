"""Sweep-5 numeric-edge regression NE-03: the "RMS in band" label states the band
actually integrated.

The read-out used to format the spinbox values with ``:.0f`` while the spinboxes
themselves accept 0.1 Hz (``setDecimals(1)``): at fs = 1 Hz the genuine 0–0.5 Hz
full band was labelled ``(0–0 Hz)`` — a zero-width band ``band_rms`` itself defines
as holding NOTHING — and at fs = 3 Hz the 0–1.5 Hz band was labelled ``(0–2 Hz)``,
an edge above Nyquist. A user typing a legal 9.5–10.5 Hz band on an ordinary 5 kHz
recording saw ``(10–10 Hz)``.

The invariant: the label restates :func:`nanoscope.analysis.psd.band_support` of the
spins — the SAME clip ``band_rms`` integrates over — at the precision the edges need,
and an empty band is named empty instead of dressed up as an integral.
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
from nanoscope.gui.psd_panel import PSDWindow


def _doc(tmp_path, name: str, fs: float, n: int = 100_000, sigma: float = 3.0,
         seed: int = 3) -> TraceDocument:
    rng = np.random.default_rng(seed)
    sig = (100.0 + rng.normal(0.0, sigma, n)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _window(qtbot, tmp_path, fs: float, **kw) -> PSDWindow:
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, f"rec-{fs:g}.dat", fs, **kw))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    return w


# ================================================================================================
# the dossier's two out-of-the-box lies
# ================================================================================================

def test_fs1_full_band_is_labelled_0_to_0p5_not_0_to_0(qapp, qtbot, tmp_path):
    w = _window(qtbot, tmp_path, fs=1.0)
    text = w.rms_readout_text()
    assert "(0–0.5 Hz)" in text, text
    assert "(0–0 Hz)" not in text


def test_fs3_full_band_is_labelled_to_its_1p5_hz_nyquist(qapp, qtbot, tmp_path):
    w = _window(qtbot, tmp_path, fs=3.0)
    text = w.rms_readout_text()
    assert "(0–1.5 Hz)" in text, text
    assert "(0–2 Hz)" not in text     # 2 Hz is above the 1.5 Hz Nyquist


def test_a_half_hz_user_band_on_an_ordinary_recording_keeps_its_edges(qapp, qtbot, tmp_path):
    """The verifier's realistic path: 9.5–10.5 Hz typed on a 5 kHz recording."""
    w = _window(qtbot, tmp_path, fs=5000.0)
    w.set_rms_band(9.5, 10.5)
    text = w.rms_readout_text()
    assert "(9.5–10.5 Hz)" in text, text
    assert "(10–10" not in text


# ================================================================================================
# the invariant itself: label == what band_rms integrated, and empty says empty
# ================================================================================================

def test_the_labelled_band_is_the_integrated_band(qapp, qtbot, tmp_path):
    """Parse the edges back out of the label; band_rms over THOSE edges must equal
    the shown value's source, rms_value(). Two surfaces, one arithmetic."""
    w = _window(qtbot, tmp_path, fs=5000.0)
    w.set_rms_band(9.5, 10.5)
    m = re.search(r"\(([\d.]+)–([\d.]+) Hz\)", w.rms_readout_text())
    assert m, w.rms_readout_text()
    lo, hi = float(m.group(1)), float(m.group(2))
    freqs, psd = w.psd_result()
    assert psd_mod.band_rms(freqs, psd, lo, hi) == pytest.approx(w.rms_value(), rel=1e-12)
    assert (lo, hi) == psd_mod.band_support(freqs, w._rms_lo_spin.value(),
                                            w._rms_hi_spin.value())


def test_a_zero_width_band_is_named_empty_not_paired_with_an_integral(qapp, qtbot, tmp_path):
    w = _window(qtbot, tmp_path, fs=5000.0)
    w.set_rms_band(10.0, 10.0)
    text = w.rms_readout_text()
    assert "empty band" in text and "nothing integrated" in text, text


def test_readout_keeps_plain_notation_for_large_nyquists(qapp, qtbot, tmp_path):
    """A 50 kHz recording's full band must read (0–25000 Hz), not (0–2.5e+04 Hz):
    the label sits beside spinboxes that show plain decimals."""
    w = _window(qtbot, tmp_path, fs=50_000.0)
    text = w.rms_readout_text()
    assert "(0–25000 Hz)" in text, text
    assert "e+" not in text


# ================================================================================================
# overlays: one shared band header only when it is true for every curve
# ================================================================================================

def test_same_rate_overlay_keeps_one_shared_band_header(qapp, qtbot, tmp_path):
    session = Session()
    a = session.add_document(_doc(tmp_path, "a.dat", 5000.0, seed=1))
    b = session.add_document(_doc(tmp_path, "b.dat", 5000.0, seed=2))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(a, 0)
    w.set_compare(["b.dat"])
    w.set_rms_band(9.5, 10.5)
    text = w.rms_readout_text()
    assert text.count("(9.5–10.5 Hz)") == 1     # once, as the shared header
    assert "a.dat" in text and "b.dat" in text
    assert b  # silence unused warning


def test_mixed_rate_overlay_states_each_curves_own_clipped_band(qapp, qtbot, tmp_path):
    """A 5 kHz and a 1 kHz pore clip the same request differently; one shared header
    would misstate one of them, so the band goes on each line."""
    session = Session()
    fast = session.add_document(_doc(tmp_path, "fast.dat", 5000.0, seed=1))
    session.add_document(_doc(tmp_path, "slow.dat", 1000.0, n=20_000, seed=2))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(fast, 0)
    w.set_compare(["slow.dat"])
    w.set_rms_band(0.0, 2500.0)
    text = w.rms_readout_text()
    assert "requested 0–2500 Hz" in text, text
    assert "(0–2500 Hz)" in text            # the fast curve integrates the full request
    assert "(0–500 Hz)" in text             # the slow curve was clipped to ITS Nyquist


def test_no_curves_still_reads_a_dash(qapp, qtbot, tmp_path):
    session = Session()
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w._update_rms_readout()
    assert w.rms_readout_text() == "—"
