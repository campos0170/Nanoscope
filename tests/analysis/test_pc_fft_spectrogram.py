"""The two spectral kernels added beside Welch: the single-shot FFT and the spectrogram.

Both live in :mod:`nanoscope.analysis.psd` next to :func:`welch_psd`, which they do not
touch — the Parseval-exact Welch estimate and every number pinned on it are unchanged, and
the load-bearing test here is precisely that the spectrogram AVERAGES BACK to it bin for
bin, so the two views can never quietly disagree about the same recording.

Synthetic float32 at 50 kHz throughout, because the answers are then exact rather than
approximate: a bin-centred sinusoid of amplitude 7.5 must read 7.5, not "about 7.5", and
that is what makes the FFT's peak-amplitude normalisation a contract rather than a habit.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import psd as psd_module
from nanoscope.analysis.psd import (
    FFT_MAX_POINTS,
    TARGET_COLUMNS,
    default_spectrogram_nperseg,
    fft_magnitude,
    spectrogram_psd,
    welch_psd,
)

FS = 50_000.0
WINDOWS = ("hann", "hamming", "blackman", "bartlett", "blackmanharris", "flattop", "boxcar")


def _noise(n: int, sigma: float = 3.0, pedestal: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, sigma, n) + pedestal).astype(np.float32)


# ================================================================================================
# (1) the single-shot FFT
# ================================================================================================

@pytest.mark.parametrize("window", WINDOWS)
def test_a_bin_centred_sinusoid_reads_its_own_amplitude_under_every_window(window):
    """The scaling is peak amplitude in signal units — under ALL seven tapers the GUI offers.

    This is why the FFT view can be labelled "|X(f)| — peak amplitude (pA)" and read with a
    ruler against the trace. The coherent-gain correction (divide by ``sum(w)``, not by
    ``n``) is what makes it exact; dividing by ``n`` would report 7.5 under boxcar and 3.75
    under hann for the identical signal.
    """
    n = 1 << 16
    k0 = 1000                                    # an exact bin: no leakage to correct for
    x = (7.5 * np.cos(2 * np.pi * k0 * np.arange(n) / n)).astype(np.float64)

    spec = fft_magnitude(x, FS, window=window, max_points=0)

    peak = int(np.argmin(np.abs(spec.freqs - k0 * FS / n)))
    assert spec.freqs[peak] == pytest.approx(k0 * FS / n)
    assert spec.mag[peak] == pytest.approx(7.5, abs=1e-6)


@pytest.mark.parametrize(
    ("window", "k0"),
    (("blackman", 1), ("blackmanharris", 1), ("flattop", 1), ("flattop", 2)),
)
@pytest.mark.parametrize("phase", np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False))
def test_near_dc_bin_amplitude_is_not_phase_biased_by_window_overlap(window, k0, phase):
    """Blackman-family mirror lobes overlap beside DC; solve them rather than guessing.

    A coherent-gain division alone made the first-bin flattop result range from 2.68 to
    12.32 for a true 7.5-unit tone depending only on its phase. The plotted amplitude
    must describe the signal, not where the acquisition happened to begin.
    """
    n = 1 << 13
    x = 7.5 * np.cos(2 * np.pi * k0 * np.arange(n) / n + phase)

    spec = fft_magnitude(x, FS, window=window, max_points=0)

    assert spec.mag[k0 - 1] == pytest.approx(7.5, abs=1e-8)


def test_the_dc_pedestal_is_removed_and_the_dc_bin_never_reaches_the_log_axis():
    """A 100 pA offset is not a frequency, and ``log10(0)`` is not a coordinate."""
    x = _noise(1 << 15, sigma=1.0, pedestal=100.0)

    spec = fft_magnitude(x, FS, max_points=0)

    assert spec.freqs.min() > 0.0                # the DC bin is gone from the drawable arrays
    assert spec.n_bins == (1 << 15) // 2         # rfft gives n/2 + 1 bins; DC dropped
    # and the pedestal did not leak into the low decade: with detrend on, the lowest
    # frequencies sit at the noise amplitude, not at 100 pA.
    assert spec.mag[:10].max() < 1.0


def test_a_mains_line_survives_the_envelope_that_fits_the_spectrum_on_a_screen():
    """The draw-side reduction is peak-preserving, and that is the whole reason for it.

    A 3 pA line at 60 Hz in a 300 000-bin spectrum reduced to 400 drawn points: the peak
    envelope keeps its full height, while the MEAN of the same geometric bin dilutes it by
    the bin's width — which is what a "reasonable" reduction would have done and why this
    test compares against that number rather than against zero.
    """
    n = 600_000
    t = np.arange(n) / FS
    x = (_noise(n, sigma=0.5, pedestal=0.0) + 3.0 * np.sin(2 * np.pi * 60.0 * t)).astype(
        np.float32)

    full = fft_magnitude(x, FS, window="boxcar", max_points=0)
    drawn = fft_magnitude(x, FS, window="boxcar", max_points=400)

    assert full.reduced is False
    assert drawn.reduced is True
    assert drawn.freqs.size <= 400 < full.n_bins
    assert drawn.n_bins == full.n_bins           # what was transformed is still reported

    j_full = int(np.argmin(np.abs(full.freqs - 60.0)))
    j_drawn = int(np.argmin(np.abs(drawn.freqs - 60.0)))
    assert full.mag[j_full] == pytest.approx(3.0, rel=0.02)
    assert drawn.mag[j_drawn] == pytest.approx(full.mag[j_full], rel=1e-12)
    # the reduced point carries the peak's OWN frequency, not its bin's centre — the bin
    # here is 3.3% wide, so a centre would have printed 59.67 Hz for a 60 Hz line
    assert drawn.freqs[j_drawn] == pytest.approx(full.freqs[j_full], rel=1e-12)
    assert drawn.freqs[j_drawn] == pytest.approx(60.0, rel=1e-4)


def test_the_default_cap_is_the_one_the_window_draws_with():
    """No hidden second policy: the module's cap is what an uncapped call applies."""
    x = _noise(4 * FFT_MAX_POINTS)
    assert fft_magnitude(x, FS).freqs.size <= FFT_MAX_POINTS


def test_a_single_nan_is_counted_rather_than_quietly_poisoning_every_bin():
    """One bad sample makes the whole transform NaN — the caller must be able to say so."""
    x = _noise(4096).astype(np.float64)
    x[123] = np.nan

    spec = fft_magnitude(x, FS, max_points=0)

    assert spec.n_nonfinite == 1
    assert not np.isfinite(spec.mag).any()       # every bin, not just the one


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_reduced_poisoned_spectrum_reports_the_problem_without_crashing(bad):
    """The screen-size envelope must survive the exact failure it is meant to display."""
    x = _noise(2 * FFT_MAX_POINTS + 2).astype(np.float64)
    x[0] = bad

    spec = fft_magnitude(x, FS)

    assert spec.reduced is True
    assert spec.n_nonfinite == 1
    assert spec.freqs.size <= FFT_MAX_POINTS
    assert not np.isfinite(spec.mag).any()


def test_a_span_too_short_or_a_dead_clock_refuses_instead_of_inventing_a_spectrum():
    with pytest.raises(ValueError, match="at least 2 samples"):
        fft_magnitude(np.array([1.0], dtype=np.float32), FS)
    with pytest.raises(ValueError, match="positive sample rate"):
        fft_magnitude(_noise(1024), 0.0)


def test_fft_working_limit_is_checked_before_the_float64_transform(monkeypatch):
    monkeypatch.setattr(psd_module, "MAX_FFT_INPUT_SAMPLES", 1_000)
    with pytest.raises(ValueError, match="above the safe.*working limit"):
        fft_magnitude(_noise(1_001), FS)


# ================================================================================================
# (2) the spectrogram
# ================================================================================================

def test_the_spectrogram_columns_average_back_to_the_welch_psd():
    """THE invariant: the spectrogram is the Welch estimate with the time average not taken.

    Same signal, same nperseg / noverlap / window / detrend — the mean over columns is the
    Welch PSD to float precision. This is what lets the two views sit on the same window
    reporting the same units and never disagree about the same recording.
    """
    x = _noise(300_000)

    freqs_w, psd_w = welch_psd(x, FS, nperseg=4096, noverlap=2048, window="hann")
    sg = spectrogram_psd(x, FS, nperseg=4096, noverlap=2048, window="hann")

    assert np.array_equal(sg.freqs, freqs_w)
    np.testing.assert_allclose(sg.psd.mean(axis=1), psd_w, rtol=1e-12, atol=0.0)


def test_spectrogram_preflights_the_unmerged_grid_not_the_screen_cap(monkeypatch):
    monkeypatch.setattr(psd_module, "MAX_SPECTROGRAM_CELLS", 1_000)
    with pytest.raises(ValueError, match="pre-merge PSD cells.*above the safe"):
        spectrogram_psd(_noise(4_096), FS, nperseg=256, noverlap=128, max_columns=2)


def test_a_burst_partway_through_is_visible_here_and_averaged_away_by_welch():
    """The reason this view exists, stated as numbers.

    A 4x-noisier stretch covering a tenth of the record lifts the WHOLE-TRACE Welch PSD by
    a factor of ~2.5 — a lift a reader would put down to a slightly noisier pore — while the
    spectrogram's own columns separate into quiet and loud by a factor of ~16, at the time
    the burst happened.
    """
    n = 500_000
    x = _noise(n, sigma=1.0, pedestal=0.0)
    lo, hi = 200_000, 250_000                     # 10% of the record, in the middle
    rng = np.random.default_rng(7)
    x[lo:hi] += rng.normal(0.0, 4.0, hi - lo).astype(np.float32)

    quiet = welch_psd(x[:lo], FS, nperseg=4096)[1]
    whole = welch_psd(x, FS, nperseg=4096)[1]
    sg = spectrogram_psd(x, FS, nperseg=4096, noverlap=2048)

    band = slice(100, None)                       # above the detrend's low-frequency corner
    lift_in_welch = float(np.median(whole[band]) / np.median(quiet[band]))
    assert 2.0 < lift_in_welch < 4.0              # a modest, undated lift

    per_column = np.median(sg.psd[band, :], axis=0)
    t_mid = (lo + hi) / 2.0 / FS
    burst_col = int(np.argmin(np.abs(sg.times - t_mid)))
    quiet_cols = sg.times < (lo / FS) * 0.9
    contrast = float(per_column[burst_col] / np.median(per_column[quiet_cols]))
    assert contrast > 10.0                        # and here it is, at the time it happened


def test_merging_columns_averages_them_rather_than_dropping_any_of_the_record():
    """A dropped column is a stretch of recording on no pixel; an averaged one is not.

    With the cap forcing a merge, the mean over the merged columns still equals the mean
    over the unmerged ones — the merge redistributes time resolution, it does not discard
    power. (Exact here because the column count divides evenly by the merge factor.)
    """
    x = _noise(400_000)
    full = spectrogram_psd(x, FS, nperseg=1024, noverlap=512, max_columns=0)
    assert full.columns_merged == 1
    assert full.times.size == 780                 # divides by 10, so the identity is exact
    capped = spectrogram_psd(x, FS, nperseg=1024, noverlap=512, max_columns=78)

    assert capped.columns_merged == 10
    assert capped.times.size == 78
    np.testing.assert_allclose(capped.psd.mean(axis=1), full.psd.mean(axis=1),
                               rtol=1e-9, atol=0.0)
    # A non-divisible count is rebinned into equal-width time cells. Boundary source
    # columns contribute fractionally, so none are dropped, the image remains affine, and
    # a short trailing group is no longer painted/weighted like a full one.
    odd = spectrogram_psd(x, FS, nperseg=1024, noverlap=512, max_columns=100)
    assert odd.columns_merged == 8 and odd.times.size == 98
    assert odd.source_columns == full.times.size
    np.testing.assert_allclose(odd.psd.mean(axis=1), full.psd.mean(axis=1),
                               rtol=2e-14, atol=0.0)
    np.testing.assert_allclose(np.diff(odd.times), odd.time_step_s,
                               rtol=2e-14, atol=1e-15)


def test_merged_time_support_uses_the_hop_not_repeated_overlapping_blocks():
    """An 8-way 50%-overlap pool spans 1024 + 7*512 samples, not 8*1024."""
    x = _noise(400_000)
    sg = spectrogram_psd(x, FS, nperseg=1024, noverlap=512, max_columns=100)

    assert sg.columns_merged == 8
    assert sg.block_duration_s == pytest.approx(1024 / FS)
    assert sg.hop_s == pytest.approx(512 / FS)
    assert sg.time_resolution_s == pytest.approx((1024 + 7 * 512) / FS)
    assert sg.time_resolution_s == pytest.approx(0.09216)
    assert sg.time_step_s == pytest.approx((sg.source_columns / sg.times.size) * 512 / FS)


def test_spectrogram_reports_the_partial_tail_scipy_does_not_analyse():
    """Requested span and complete-window support are related, but not identical."""
    n = 10_123
    nperseg = 1024
    noverlap = 512
    hop = nperseg - noverlap
    sg = spectrogram_psd(
        _noise(n), FS, nperseg=nperseg, noverlap=noverlap, max_columns=7)

    expected_analysed = nperseg + ((n - nperseg) // hop) * hop
    assert sg.analyzed_samples == expected_analysed
    assert sg.uncovered_tail_samples == n - expected_analysed
    assert 0 <= sg.uncovered_tail_samples < hop
    # Display rebinning changes columns, never the complete-window support.
    assert sg.source_columns > sg.times.size


def test_the_spectrogram_auto_targets_columns_where_the_welch_auto_targets_averages():
    """Two autos, two questions. A spectrogram with 8 columns has no time axis to read."""
    n = 2_000_000
    nps = default_spectrogram_nperseg(n)
    assert n // nps >= TARGET_COLUMNS // 2        # hundreds of blocks, not eight
    sg = spectrogram_psd(_noise(n), FS, max_columns=0)
    assert sg.times.size >= TARGET_COLUMNS


def test_the_time_axis_is_measured_from_the_first_sample_of_the_span():
    """The kernel owns no clock: times start at the span, and the caller adds its offset.

    The window adds ``i0 / fs`` so the drawn axis is in the SAME recording-seconds the
    Segment spins take — the clock the app's own report says a 5.7x error was typed into.
    """
    sg = spectrogram_psd(_noise(200_000), FS, nperseg=4096, noverlap=2048)
    assert sg.times[0] == pytest.approx(4096 / 2 / FS, rel=1e-9)
    assert sg.times[-1] < 200_000 / FS


def test_a_spectrogram_of_a_span_never_sees_more_than_the_span():
    """Slicing happens before the transform, so a 2 s look costs 2 s of samples.

    A 12 M-sample recording is not what a 2 s view transforms: the kernel's own report of
    ``n_samples`` is the span's length, and it is the only array it was given.
    """
    big = _noise(3_000_000, seed=11)
    span = big[100_000:200_000]                   # 2 s at 50 kHz — a VIEW, not a copy
    sg = spectrogram_psd(span, FS, max_columns=900)
    assert sg.n_samples == 100_000
    assert span.base is big                       # no copy was made to get here
