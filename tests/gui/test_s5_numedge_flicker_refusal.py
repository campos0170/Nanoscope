"""Sweep-5 numeric-edge regression NE-04: a flicker refusal names a band that exists,
or says WHY none does at this sample rate.

``noise_summary`` clips the fixed 1–100 Hz flicker band's edges independently — the
low edge up to the spectrum's first bin, the high edge down to Nyquist — so at
fs = 1 Hz the pair CROSSES to (1.0, 0.5) and at fs = 250 MHz to (3815, 100). The
Details pane then printed "the 1–0.5 Hz flicker band holds 0 FFT bins..." — a band
that does not exist, rendered as if it did, with a remedy (raise nperseg) that at
fs = 1 Hz cannot help at all.

The refusal now detects the crossed pair and names the real cause, each with the
remedy that actually applies:

* Nyquist below the band -> no flicker band exists at this fs; no Welch setting helps.
* Δf coarser than the whole band -> raise the segment length until Δf fits.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import PSDWindow

_BAND = re.compile(r"the ([\d.e+\-]+)–([\d.e+\-]+) Hz flicker band")


def _details_for(qtbot, tmp_path, fs: float, n: int = 1_000_000) -> str:
    rng = np.random.default_rng(3)
    sig = (100.0 + rng.normal(0.0, 3.0, n)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / f"r-{fs:g}.dat", format_name="synthetic",
                         n_samples=n, sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(sig,)))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    # the Details pane hard-wraps to its column width; fold the wrapping back out so
    # the assertions read phrases, not line-break placement
    return " ".join(w.details_text().split())


def _no_inverted_band(text: str) -> None:
    """No surface may print an inverted pair as a band, anywhere in the details."""
    for m in _BAND.finditer(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        assert hi > lo, f"inverted band rendered as real: {m.group(0)!r}"


def test_fs1_names_nyquist_below_the_band_not_an_inverted_pair(qapp, qtbot, tmp_path):
    text = _details_for(qtbot, tmp_path, fs=1.0)
    _no_inverted_band(text)
    assert "1–0.5" not in text
    assert "no flicker band exists" in text
    assert "Nyquist (0.5 Hz)" in text
    # the honest remedy: no Welch setting reaches a band above Nyquist
    assert "No Welch setting can reach it" in text
    assert "sample rate above 2 Hz" in text


def test_fs250mhz_names_the_coarse_resolution_not_an_inverted_pair(qapp, qtbot, tmp_path):
    text = _details_for(qtbot, tmp_path, fs=250e6)
    _no_inverted_band(text)
    assert "3815–100" not in text
    assert "Δf = 3815 Hz is coarser" in text
    # here the segment-length remedy IS the right one, with the concrete target
    assert "Raise the Welch segment length" in text
    assert "Δf ≤ 100 Hz" in text


def test_an_ordinary_recording_still_reports_its_exponent(qapp, qtbot, tmp_path):
    """The refusal branches must not swallow the working case: at 50 kHz over 1 M
    samples the 1–100 Hz band holds plenty of bins and α is printed with them."""
    text = _details_for(qtbot, tmp_path, fs=50_000.0)
    _no_inverted_band(text)
    assert "1/f exponent α: unavailable" not in text     # text is whitespace-folded
    m = re.search(r"fit over ([\d.]+)–([\d.]+) Hz on (\d+) FFT bins", text)
    assert m, text
    assert float(m.group(2)) > float(m.group(1))
    assert int(m.group(3)) >= 4


def test_a_band_that_exists_but_is_too_coarse_keeps_the_bin_count_refusal(qapp, qtbot, tmp_path):
    """Between the crossed cases sits the original refusal — a real band with too few
    bins — which must still name the band, its count and the fit's floor."""
    # fs = 12.8 kHz over 2048 samples -> nperseg 256, Δf = 50 Hz: the clipped flicker
    # band is a REAL 50-100 Hz with only 2 bins — under the fit's floor of 4.
    text = _details_for(qtbot, tmp_path, fs=12_800.0, n=2048)
    _no_inverted_band(text)
    m = _BAND.search(text)
    assert m, text
    assert float(m.group(2)) > float(m.group(1))       # a band that exists
    assert "unavailable" in text
    m2 = re.search(r"holds (\d+) FFT bins? and this fit needs 4", text)
    assert m2, text
    assert int(m2.group(1)) < 4


def test_the_crossed_pair_stays_available_to_machine_readers(qapp, qtbot, tmp_path):
    """The summary dict still carries the per-edge clips (documented as crossable) —
    the fix is in the RENDERING, not a silent rewrite of the data."""
    from nanoscope.analysis import psd as psd_mod
    x = np.random.default_rng(0).normal(0.0, 1.0, 4096)
    freqs, psd = psd_mod.welch_psd(x, 1.0)
    s = psd_mod.noise_summary(freqs, psd, 1.0)
    lo, hi = s["flicker_band"]
    assert lo >= hi                       # crossed: the clips emptied the band
    assert s["flicker"]["status"] == "unavailable"
    assert not np.isfinite(s["flicker"]["alpha"])
