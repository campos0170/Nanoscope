"""The PSD window's CAPTIONS as claims, and the export they must not shrink.

Five defects found after the four-view window landed, each of them a sentence or a number
that could be checked and was wrong. This file checks them:

* **PC-REGRESS / PC-C1** — the PSD page reserved a 96 px caption box that never filled. It
  cost the plot a fifth of its height and, because "Save image…" exports the PlotItem at
  its NATIVE size, it silently shortened every exported spectrum by 100 px (a 691x488 PNG
  became 691x388). The page is now the bare plot widget; the export is the one it was.
* **PC-SPECCAP** — the spectrogram caption asserted its columns average to the PSD view
  "bin for bin". True only at the same block length and hop, and the two views ship with
  DIFFERENT defaults (512 against 16 384 on a 5 s span). The claim is now conditional, and
  this file measures BOTH states: 0.0 max relative difference where it is made, and a
  bin-count mismatch where it is withheld.
* **PC-SPECASD** — the same caption said "same units as the PSD view" while that view can
  be drawing the ASD in pA/√Hz against this image's pA²/Hz.
* **PC-LEVELPCT** — the all-points level readout printed ONE BIN's share as the fraction of
  time the pore spent at that level: 6.6% at 128 bins and 0.21% at 4 096 for the SAME level
  that holds 99.2% of the record. The percentage is now integrated over the bins nearest
  that level (the half-amplitude rule) and is stable in the bin count.
* **PC-C3 / PC-FFTVAR** — the reduced FFT's max envelope draws the noise floor high and the
  caption did not say so, while it did quote the power periodogram's ~100% uncertainty on
  an axis drawing magnitude (~52%). Both numbers are re-measured here from the kernels, so
  the caption's arithmetic is pinned and not merely spell-checked.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QImage

from nanoscope.analysis import psd as psd_mod
from nanoscope.analysis.allpoints import all_points_histogram, level_peaks
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import (
    VIEW_ALLPOINTS,
    VIEW_FFT,
    VIEW_PSD,
    VIEW_SPECTROGRAM,
    PSDWindow,
    _level_occupancy,
)

FS = 50_000.0
N = 250_000
OPEN_PA, BLOCKED_PA = 100.0, 60.0
BLOCK_FROM, BLOCK_TO = 60_000, 62_000          # 2 000 samples: 0.8% of the record
TRUE_BLOCKED = (BLOCK_TO - BLOCK_FROM) / N     # 0.008 exactly, by construction


def _pore_trace(seed: int = 0) -> np.ndarray:
    """5 s at 100 pA with a 2 000-sample blockade to 60 pA and a 60 Hz mains line."""
    rng = np.random.default_rng(seed)
    sig = (rng.normal(0.0, 3.0, N) + OPEN_PA).astype(np.float32)
    sig += (2.0 * np.sin(2 * np.pi * 60.0 * np.arange(N) / FS)).astype(np.float32)
    sig[BLOCK_FROM:BLOCK_TO] -= (OPEN_PA - BLOCKED_PA)
    return sig


@pytest.fixture
def psd(qapp, qtbot, tmp_path):
    """The PSD window on one synthetic pore recording, at its own default size."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "pore.dat", format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(_pore_trace(),)))
    win = PSDWindow(session)
    qtbot.addWidget(win)
    win.resize(1080, 720)
    win.show()
    win.set_target(doc_id, 0)
    qapp.processEvents()
    return win


# ================================================================================================
# PC-REGRESS / PC-C1 — the box that never filled, and the export it shortened
# ================================================================================================

def test_the_psd_page_reserves_no_caption_box_and_keeps_its_whole_plot(psd, qapp):
    """No empty box under the spectrum: the PSD page IS the plot widget.

    Pre-fix the window built four identical pages — plot above a fixed 96 px caption — and
    only three of them ever wrote text into one. This asserts the fourth is not built, and
    that the plot is taller by at least that box as a result.
    """
    assert VIEW_PSD not in psd._captions, "the PSD page must not reserve a caption box"
    assert psd.caption_text(VIEW_PSD) == ""
    assert set(psd._captions) == {VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS}

    psd.set_view(VIEW_PSD)
    qapp.processEvents()
    tall = psd._plot_widget.height()
    psd.set_view(VIEW_FFT)
    qapp.processEvents()
    short = psd._fft_widget.height()
    box = psd._captions[VIEW_FFT].height()
    assert box > 0
    assert tall - short >= box, (
        f"the PSD plot ({tall} px) should keep the {box} px a caption box would take "
        f"from it; the FFT plot beside it is {short} px")


def test_saving_the_psd_image_writes_the_plots_own_native_size(psd, qapp, tmp_path):
    """The pre-existing export, measured: PNG height == the plot widget's height.

    This is the shape of PC-REGRESS. The exporter has no size of its own at the shipped
    ``export_w/h = 0``; it renders the PlotItem as it stands, so anything that shortens the
    widget shortens every spectrum the owner has ever exported. Pinning the identity here
    means the next thing added under the plot fails this test instead of the PNG.
    """
    psd.set_view(VIEW_PSD)
    qapp.processEvents()
    out = psd.save_image(str(tmp_path / "noise_psd.png"))
    assert out is not None and out.exists()
    img = QImage(str(out))
    assert img.height() == psd._plot_widget.height()
    assert img.width() == psd._plot_widget.width()


# ================================================================================================
# PC-SPECCAP / PC-SPECASD — a conditional claim, checked in both states
# ================================================================================================

def _spec_caption(psd, qapp) -> str:
    psd.set_view(VIEW_SPECTROGRAM)
    qapp.processEvents()
    return psd.caption_text(VIEW_SPECTROGRAM)


def test_the_spectrogram_withholds_the_bin_for_bin_claim_at_the_shipped_defaults(psd, qapp):
    """At the defaults the two views do not even have the same number of bins."""
    cap = _spec_caption(psd, qapp)
    sg = psd.spectrogram()
    assert sg.nperseg != psd._nperseg_used, (
        "this test is about the state where the two block lengths differ")
    assert sg.freqs.size != psd._freqs.size
    assert "does NOT return that spectrum" in cap
    # and it names both numbers the reader would need to check it
    assert f"{sg.nperseg:,} samples" in cap
    assert f"{psd._nperseg_used:,}-sample Welch segment" in cap
    # the advice it gives must be a setting the window actually offers
    assert f"Set Time resolution to {psd._nperseg_used:,}" in cap
    assert psd._spec_nperseg_combo.findText(str(psd._nperseg_used)) >= 0


def test_the_bin_for_bin_claim_appears_only_when_it_is_exactly_true(psd, qapp):
    """Matched blocks: the caption makes the claim AND the arrays agree to the bit."""
    _spec_caption(psd, qapp)
    psd._spec_nperseg_combo.setCurrentText(str(psd._nperseg_used))
    qapp.processEvents()
    cap = psd.caption_text(VIEW_SPECTROGRAM)
    sg = psd.spectrogram()
    assert sg.nperseg == psd._nperseg_used and sg.columns_merged == 1
    assert "returns that spectrum bin for bin" in cap
    assert "does NOT return" not in cap

    column_mean = sg.psd.mean(axis=1)
    assert column_mean.size == psd._psd.size
    rel = np.abs(column_mean - psd._psd) / np.abs(psd._psd)
    assert float(rel.max()) == 0.0, (
        f"the caption claims bin-for-bin agreement; measured max relative difference "
        f"{rel.max():.3g}")


def test_the_spectrogram_says_when_its_units_differ_from_the_psd_views(psd, qapp):
    """ASD ticked: that view draws pA/√Hz while this colour bar draws pA²/Hz."""
    psd.set_asd(False)
    cap_off = _spec_caption(psd, qapp)
    assert "√Hz" not in cap_off

    psd.set_view(VIEW_PSD)
    psd.set_asd(True)
    qapp.processEvents()
    cap_on = _spec_caption(psd, qapp)
    assert "pA/√Hz" in cap_on and "pA²/Hz" in cap_on
    assert "same units" not in cap_on
    # and the window really is in that state, not merely captioned for it
    assert psd.is_asd()
    assert "ASD" in psd._plot.getAxis("left").labelText
    assert "log₁₀ PSD (pA²/Hz)" == psd._spec_bar.getAxis("right").labelText


# ================================================================================================
# PC-LEVELPCT — an occupancy the histogram can support
# ================================================================================================

@pytest.mark.parametrize("bins", ["128", "512", "2048"])
def test_the_level_percentages_are_occupancy_not_one_bins_share(psd, qapp, bins):
    """0.8% of this record is blocked, and the caption must say so at every binning."""
    psd.set_view(VIEW_ALLPOINTS)
    psd._hist_bins_combo.setCurrentText(bins)
    qapp.processEvents()
    levels = psd.hist_levels()
    assert len(levels) >= 2, "the synthetic has two levels by construction"
    hist = psd.all_points()
    occ = _level_occupancy(hist, levels)
    assert occ[0] == pytest.approx(1.0 - TRUE_BLOCKED, abs=5e-4)
    assert occ[1] == pytest.approx(TRUE_BLOCKED, abs=5e-4)
    assert sum(occ) == pytest.approx(1.0, abs=1e-9)

    line = next(l for l in psd.caption_text().splitlines() if l.startswith("Levels"))
    assert "99.2%" in line and "0.8%" in line
    # the one-bin share the readout used to print, for the same level:
    one_bin = levels[0]["fraction"]
    assert one_bin < 0.5, "the peak BIN never holds the occupancy on a fine binning"
    assert "INTEGRATES" in line and "half-amplitude" in line
    assert "boundaries at" in line          # it says what window it integrated over


def test_one_bins_share_moves_with_the_binning_and_the_occupancy_does_not(qapp):
    """The measurement behind PC-LEVELPCT, with no window in the way."""
    sig = _pore_trace()
    one_bin, integrated = [], []
    for bins in (128, 256, 512, 1024, 2048):
        hist = all_points_histogram(sig, bins=bins)
        levels = level_peaks(hist)
        assert len(levels) >= 2
        one_bin.append(levels[0]["fraction"])
        integrated.append(_level_occupancy(hist, levels)[0])
    assert max(one_bin) / min(one_bin) > 8, (
        "the old percentage is a property of the bin count: it should swing wildly")
    assert max(integrated) - min(integrated) < 1e-3, (
        f"the integrated occupancy must not depend on the binning: {integrated}")
    assert all(abs(v - (1.0 - TRUE_BLOCKED)) < 5e-4 for v in integrated)


# ================================================================================================
# PC-C3 / PC-FFTVAR — the floor the envelope lifts, and the scatter of what is drawn
# ================================================================================================

def test_the_fft_caption_declares_that_its_envelope_draws_the_floor_high(psd, qapp):
    psd.set_view(VIEW_FFT)
    qapp.processEvents()
    spec = psd.fft_spectrum()
    assert spec.reduced, "a 250 000-sample span is reduced at the shipped max_points"
    cap = psd.caption_text(VIEW_FFT)
    assert "FLOOR" in cap and "HIGHER than it is" in cap
    assert "Read a NOISE FLOOR from the Welch PSD view" in cap
    assert "LARGEST of the" in cap            # and it says what the reduction does
    assert "envelope" in cap.lower()


def test_the_fft_caption_quotes_the_scatter_of_the_axis_it_draws(psd, qapp):
    """~100% is the POWER periodogram's figure; this axis draws magnitude (~52%)."""
    psd.set_view(VIEW_FFT)
    qapp.processEvents()
    cap = psd.caption_text(VIEW_FFT)
    assert "52%" in cap
    assert "100% uncertainty" not in cap
    assert "in power" in cap                  # the 100% figure is kept, correctly attributed
    assert "periodogram" in cap.lower() and "unaveraged" in cap.lower()


def test_the_two_numbers_that_caption_quotes_are_the_measured_ones():
    """Re-measure both claims from the kernels, so the caption is pinned to arithmetic."""
    rng = np.random.default_rng(11)

    # (a) the max envelope lifts the drawn floor, and by more the harsher the reduction.
    lifts = {}
    for n in (200_000, 1_000_000):
        y = rng.normal(0.0, 3.0, n).astype(np.float32)
        full = psd_mod.fft_magnitude(y, FS, window="hann", max_points=0)
        drawn = psd_mod.fft_magnitude(y, FS, window="hann")
        assert drawn.reduced and not full.reduced
        per_point = full.n_bins / drawn.freqs.size
        lifts[round(per_point)] = float(np.median(drawn.mag) / np.median(full.mag))
    assert min(lifts) == pytest.approx(11, abs=2) and max(lifts) == pytest.approx(48, abs=4)
    assert lifts[min(lifts)] == pytest.approx(1.7, abs=0.15)
    assert lifts[max(lifts)] == pytest.approx(2.0, abs=0.15)

    # (b) the scatter of the DRAWN quantity is Rayleigh's 52%, not the power's 100%.
    cv_mag, cv_pow = [], []
    for _ in range(8):
        y = rng.normal(0.0, 5.0, 100_000).astype(np.float32)
        m = psd_mod.fft_magnitude(y, FS, window="boxcar", max_points=0).mag[1:-1]
        cv_mag.append(float(m.std() / m.mean()))
        cv_pow.append(float((m ** 2).std() / (m ** 2).mean()))
    assert np.mean(cv_mag) == pytest.approx(0.52, abs=0.03)
    assert np.mean(cv_pow) == pytest.approx(1.0, abs=0.05)
