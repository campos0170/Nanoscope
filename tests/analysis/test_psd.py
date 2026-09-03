"""Tests for analysis.psd: ground-truth signals in, known noise numbers out.

Everything is checked against an analytic truth with a SEEDED generator
(``np.random.default_rng(0)``), never a bare ``np.random`` call:

* **Parseval / RMS** — white Gaussian noise of variance ``sigma**2`` integrates
  (over the whole band) to an RMS of ``sigma``; a pure tone of amplitude ``A``
  puts a peak in the right bin and integrates (over a band bracketing it) to
  ``A/sqrt(2)``.
* **1/f fit** — noise built with a known PSD slope ``f**(-alpha)`` is recovered
  by :func:`fit_flicker` to within tolerance, with a positive ``A1Hz`` and a
  high log-space ``R^2``; a white band fits ``alpha ~ 0``.
* **noise_summary** — the per-band RMS values add in quadrature to the total
  (within windowing leakage), the white floor matches the flat one-sided level
  ``2*sigma**2/fs``, and bands lying above a low Nyquist collapse to ``0.0`` (not
  NaN).

The units are exercised implicitly: the kernels are unit-agnostic, so a signal
in pA yields a PSD in pA**2/Hz and an RMS in pA — the numbers below are in
whatever unit the input array carries.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.psd import (
    DEFAULT_BANDS,
    FLICKER_BAND_HZ,
    FLICKER_MARGINAL_POINTS,
    FLICKER_MIN_POINTS,
    asd_from_psd,
    band_rms,
    default_nperseg,
    drop_dc,
    fit_flicker,
    flicker_status,
    noise_summary,
    welch_psd,
)


# ================================================================================================
# builders
# ================================================================================================

def _white(n: int, sigma: float, *, seed: int = 0) -> np.ndarray:
    """White Gaussian noise, zero mean, standard deviation ``sigma`` (seeded)."""
    return np.random.default_rng(seed).normal(0.0, sigma, size=n)


def _one_over_f(n: int, fs: float, alpha: float, *, seed: int = 0) -> np.ndarray:
    """Noise whose expected PSD follows ``f**(-alpha)`` (a shaped-white construction).

    White noise is coloured in the frequency domain by ``|H(f)| = f**(-alpha/2)``
    so ``|H|**2 = f**(-alpha)``; the DC bin is zeroed (no ``f=0`` for a power law).
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    h = np.zeros_like(f)
    h[1:] = f[1:] ** (-alpha / 2.0)
    return np.fft.irfft(spec * h, n=n)


# ================================================================================================
# (a) Parseval / RMS — white noise variance and a pure tone
# ================================================================================================

def test_band_rms_recovers_white_noise_sigma():
    """band_rms over [0, Nyquist] of white noise of variance sigma**2 ~= sigma."""
    fs = 50_000.0
    sigma = 2.5
    x = _white(1 << 19, sigma)                       # ~10 s, plenty for averaging
    freqs, psd = welch_psd(x, fs)
    rms = band_rms(freqs, psd, 0.0, fs / 2.0)
    assert rms == pytest.approx(sigma, rel=0.03)     # within a few percent


def test_psd_units_scale_with_the_signal():
    """A signal scaled by c has a PSD scaled by c**2 and an RMS scaled by c (pA vs nA)."""
    fs = 50_000.0
    x = _white(1 << 18, 3.0)
    f1, p1 = welch_psd(x, fs)
    f2, p2 = welch_psd(1e-3 * x, fs)                 # e.g. the same trace read in nA not pA
    np.testing.assert_allclose(p2, p1 * 1e-6, rtol=1e-9, atol=0.0)
    assert band_rms(f2, p2, 0.0, fs / 2.0) == pytest.approx(
        1e-3 * band_rms(f1, p1, 0.0, fs / 2.0), rel=1e-6)


def test_pure_tone_peaks_in_the_right_bin_and_integrates_to_rms():
    """A tone A*sin(2*pi*f0*t) peaks at f0 and band_rms around it ~= A/sqrt(2)."""
    fs = 50_000.0
    nperseg = 10_000                                 # df = 5 Hz -> f0=1000 lands on a bin
    A = 5.0
    f0 = 1_000.0
    t = np.arange(1 << 19) / fs
    x = A * np.sin(2.0 * np.pi * f0 * t)
    freqs, psd = welch_psd(x, fs, nperseg=nperseg, noverlap=nperseg // 2)

    peak_hz = float(freqs[int(np.argmax(psd))])
    assert peak_hz == pytest.approx(f0, abs=2.0 * fs / nperseg)   # within a couple of bins

    rms = band_rms(freqs, psd, f0 - 40.0, f0 + 40.0)
    assert rms == pytest.approx(A / np.sqrt(2.0), rel=0.05)

    # essentially all the power is in the tone: a band away from it is near-silent
    away = band_rms(freqs, psd, 5_000.0, 10_000.0)
    assert away < 0.02 * rms


def test_band_rms_zero_for_empty_or_above_nyquist_band():
    fs = 20_000.0
    freqs, psd = welch_psd(_white(1 << 16, 1.0), fs)
    assert band_rms(freqs, psd, 0.0, 0.0) == 0.0                 # degenerate band
    assert band_rms(freqs, psd, 50_000.0, 80_000.0) == 0.0       # wholly above Nyquist
    assert band_rms(freqs, psd, 8_000.0, 5_000.0) == 0.0         # inverted band


# ================================================================================================
# (b) synthetic 1/f — recover the slope and the 1 Hz magnitude
# ================================================================================================

@pytest.mark.parametrize("alpha", [0.7, 1.0, 1.4])
def test_fit_flicker_recovers_known_slope(alpha):
    fs = 50_000.0
    x = _one_over_f(1 << 19, fs, alpha)
    freqs, psd = welch_psd(x, fs, nperseg=8192)
    fit = fit_flicker(freqs, psd, f_lo=12.0, f_hi=2_000.0)

    assert fit["alpha"] == pytest.approx(alpha, abs=0.15)        # exponent recovered
    assert fit["A1Hz"] > 0.0 and np.isfinite(fit["A1Hz"])        # a positive PSD magnitude
    assert fit["r2"] > 0.8                                       # a clean power law
    assert fit["n"] > 10


def test_fit_flicker_flat_band_is_alpha_near_zero():
    """A white (flat) band fits to an exponent near zero — no 1/f rise."""
    fs = 50_000.0
    freqs, psd = welch_psd(_white(1 << 19, 1.5), fs, nperseg=8192)
    fit = fit_flicker(freqs, psd, f_lo=2_000.0, f_hi=20_000.0)
    assert abs(fit["alpha"]) < 0.15


def test_fit_flicker_unavailable_is_nan_not_raised():
    """A band with too few points reports NaNs and n=0 rather than raising."""
    fs = 50_000.0
    freqs, psd = welch_psd(_white(1 << 16, 1.0), fs)
    fit = fit_flicker(freqs, psd, f_lo=1_000.0, f_hi=1_000.0)    # empty band
    assert fit["n"] == 0
    assert np.isnan(fit["alpha"]) and np.isnan(fit["A1Hz"]) and np.isnan(fit["r2"])


# ================================================================================================
# (c) noise_summary — bands add in quadrature, white floor, low-fs collapse
# ================================================================================================

def test_noise_summary_bands_sum_in_quadrature_to_total():
    fs = 50_000.0                                    # Nyquist 25 kHz -> all three bands live
    sigma = 4.0
    x = _white(1 << 19, sigma)
    freqs, psd = welch_psd(x, fs)
    summary = noise_summary(freqs, psd, fs)

    total = summary["total_rms"]
    assert total == pytest.approx(sigma, rel=0.03)

    quad = np.sqrt(sum(v ** 2 for v in summary["bands"].values()))
    assert quad == pytest.approx(total, rel=0.02)    # partition of [0, Nyquist]


def test_noise_summary_white_floor_matches_flat_level():
    """The white floor ~= the one-sided flat density 2*sigma**2/fs for white noise."""
    fs = 50_000.0
    sigma = 3.0
    freqs, psd = welch_psd(_white(1 << 19, sigma), fs)
    summary = noise_summary(freqs, psd, fs)
    expected_flat = 2.0 * sigma ** 2 / fs
    assert summary["white_floor"] == pytest.approx(expected_flat, rel=0.15)
    lo, hi = summary["white_floor_band"]
    assert 0.0 < lo < hi <= fs / 2.0


def test_noise_summary_bands_above_nyquist_are_zero_not_nan():
    fs = 2_000.0                                     # Nyquist 1 kHz: the upper bands vanish
    freqs, psd = welch_psd(_white(1 << 16, 1.0), fs)
    summary = noise_summary(freqs, psd, fs)
    bands = summary["bands"]

    assert bands["1-10 kHz"] == 0.0
    assert bands["10 kHz-Nyquist"] == 0.0
    for v in bands.values():
        assert np.isfinite(v)                        # zero, never NaN
    # the only surviving band is the sub-kHz one, which spans the whole [0, Nyq]
    assert bands["<1 kHz"] == pytest.approx(summary["total_rms"], rel=1e-6)


def test_noise_summary_reports_resolution_and_nyquist():
    fs = 50_000.0
    nperseg = 8192
    freqs, psd = welch_psd(_white(1 << 18, 1.0), fs, nperseg=nperseg)
    summary = noise_summary(freqs, psd, fs)
    assert summary["nyquist_hz"] == pytest.approx(fs / 2.0, rel=1e-9)
    assert summary["df_hz"] == pytest.approx(fs / nperseg, rel=1e-6)
    # the built-in flicker fit runs over the fixed FLICKER_BAND_HZ, clipped to
    # [first positive bin, Nyquist] — NOT the nperseg-dependent lowest decade
    fl_lo, fl_hi = summary["flicker_band"]
    assert fl_lo == pytest.approx(max(FLICKER_BAND_HZ[0], summary["df_hz"]))
    assert fl_hi == pytest.approx(min(FLICKER_BAND_HZ[1], summary["nyquist_hz"]))


def test_noise_summary_flicker_stable_under_large_nperseg():
    """Regression: a large nperseg on a short record used to collapse the auto
    flicker band ([f1, 10*f1]) onto a few noisy bins and report a spurious NEGATIVE
    alpha the GUI still drew as a rising "1/f" line. With the band pinned to
    FLICKER_BAND_HZ the exponent stays sane regardless of segment length."""
    fs = 50_000.0
    x = _one_over_f(int(2 * fs), fs, alpha=1.0)          # a short 2 s record of genuine 1/f
    for nps in (8192, 65536):                            # Auto-ish vs the pathological large one
        summary = noise_summary(*welch_psd(x, fs, nperseg=nps), fs)
        fl = summary["flicker"]
        # fit sits in the pinned band, not the collapsing lowest decade
        assert fl["f_hi"] == pytest.approx(min(FLICKER_BAND_HZ[1], summary["nyquist_hz"]))
        assert fl["f_lo"] <= 10.0
        # a positive, physical exponent both ways — never the old negative-alpha garbage
        assert fl["alpha"] > 0.3
        assert np.isfinite(fl["A1Hz"]) and fl["A1Hz"] > 0.0


def test_noise_summary_flicker_unavailable_when_too_few_low_freq_bins():
    """A coarse-resolution spectrum (few bins below 100 Hz) reports the 1/f fit as
    unavailable (NaN) rather than trusting a 2-3 point slope."""
    fs = 50_000.0
    # nperseg 512 -> df ~97.7 Hz, so at most ~1 bin sits inside [1, 100] Hz
    freqs, psd = welch_psd(_white(1 << 16, 1.0), fs, nperseg=512)
    fl = noise_summary(freqs, psd, fs)["flicker"]
    assert np.isnan(fl["alpha"]) and np.isnan(fl["A1Hz"])


# ================================================================================================
# (d) how much the 1/f exponent RESTS ON — report §7 Medium-14
# ================================================================================================

def test_flicker_status_is_the_single_arbiter():
    """One place decides what a bin count is worth; the GUI never re-derives it."""
    assert FLICKER_MIN_POINTS == 4 and FLICKER_MARGINAL_POINTS == 8
    assert flicker_status(0) == "unavailable"
    assert flicker_status(FLICKER_MIN_POINTS - 1) == "unavailable"
    assert flicker_status(FLICKER_MIN_POINTS) == "marginal"       # AT the floor, not above it
    assert flicker_status(FLICKER_MARGINAL_POINTS - 1) == "marginal"
    assert flicker_status(FLICKER_MARGINAL_POINTS) == "ok"
    assert flicker_status(400) == "ok"


def test_fit_flicker_reports_the_bin_count_it_used():
    """An exponent is only as good as the number of bins under it, and R² cannot
    say whether that number is 4 or 400 — so ``n`` and the verdict travel WITH it."""
    fs = 50_000.0
    freqs, psd = welch_psd(_one_over_f(1 << 19, fs, 1.0), fs, nperseg=8192)
    fit = fit_flicker(freqs, psd, f_lo=1.0, f_hi=100.0)
    assert fit["n"] >= FLICKER_MARGINAL_POINTS
    assert fit["n_band"] >= fit["n"]                  # bins in the band vs bins fitted
    assert fit["status"] == flicker_status(fit["n"])
    assert fit["status"] == "ok"


def test_fit_flicker_at_the_hard_minimum_is_marginal_not_confident():
    """The −600 mV geometry exactly: a 0.4 s quiet segment, Welch nperseg 2048, and
    the fixed 1-100 Hz band holding FOUR bins. The exponent is still returned — it is
    real — but it may never be presented as if it were well supported."""
    fs = 50_000.0
    freqs, psd = welch_psd(_one_over_f(19_915, fs, 1.0), fs, nperseg=2048)
    assert freqs[1] == pytest.approx(fs / 2048, rel=1e-6)          # Δf ≈ 24.41 Hz
    fit = fit_flicker(freqs, psd, f_lo=freqs[1], f_hi=100.0)
    assert fit["n"] == FLICKER_MIN_POINTS == 4
    assert fit["status"] == "marginal"
    assert np.isfinite(fit["alpha"]) and np.isfinite(fit["r2"])


def test_noise_summary_marginal_flicker_keeps_alpha_and_flags_it():
    fs = 50_000.0
    fl = noise_summary(*welch_psd(_one_over_f(19_915, fs, 1.0), fs, nperseg=2048), fs)["flicker"]
    assert fl["n"] == 4 and fl["status"] == "marginal"
    assert np.isfinite(fl["alpha"])                   # blanking a real fit would be a lie too


def test_noise_summary_unavailable_flicker_still_says_how_many_bins_it_found():
    """"Too few" is not a number. The count survives the blanking, because it is how
    a caller explains WHY there is no exponent and how much longer a segment needs."""
    fs = 50_000.0
    freqs, psd = welch_psd(_white(1 << 16, 1.0), fs, nperseg=512)   # Δf ≈ 97.7 Hz
    fl = noise_summary(freqs, psd, fs)["flicker"]
    assert fl["status"] == "unavailable" and np.isnan(fl["alpha"])
    assert 0 < fl["n_band"] < FLICKER_MIN_POINTS


# ================================================================================================
# guards + small helpers
# ================================================================================================

def test_welch_psd_guards_empty_and_bad_fs():
    with pytest.raises(ValueError):
        welch_psd(np.zeros(0), 50_000.0)
    with pytest.raises(ValueError):
        welch_psd(np.zeros(1), 50_000.0)             # single sample: too short
    with pytest.raises(ValueError):
        welch_psd(np.zeros(1000), 0.0)               # non-positive fs
    with pytest.raises(ValueError):
        welch_psd(np.zeros(1000), float("nan"))


def test_welch_psd_default_nperseg_is_bounded_and_averaging():
    n = 1 << 20
    nps = default_nperseg(n)
    assert (nps & (nps - 1)) == 0                    # a power of two
    assert 256 <= nps <= (1 << 16)
    assert n // nps >= 8                             # at least ~8 non-overlapping segments
    # a short record uses the whole thing as one segment
    assert default_nperseg(100) == 100


def test_welch_psd_honours_explicit_nperseg_clamped_to_length():
    x = _white(5_000, 1.0)
    freqs, psd = welch_psd(x, 10_000.0, nperseg=99_999)   # larger than the signal
    # scipy would warn+clamp; we clamp to len(x), giving a single-segment periodogram grid
    assert freqs.size == x.size // 2 + 1
    assert np.all(np.isfinite(psd))


def test_drop_dc_removes_the_zero_bin():
    freqs, psd = welch_psd(_white(1 << 15, 1.0), 10_000.0)
    assert freqs[0] == 0.0                            # a one-sided spectrum has a DC bin
    f2, p2 = drop_dc(freqs, psd)
    assert f2.size == freqs.size - 1
    assert np.all(f2 > 0.0)
    np.testing.assert_array_equal(p2, psd[1:])


def test_asd_from_psd_is_sqrt_and_nan_safe():
    psd = np.array([4.0, 9.0, 0.0, -1.0, np.nan])
    asd = asd_from_psd(psd)
    assert asd[0] == pytest.approx(2.0)
    assert asd[1] == pytest.approx(3.0)
    assert asd[2] == pytest.approx(0.0)
    assert np.isnan(asd[3]) and np.isnan(asd[4])     # negative / NaN -> NaN, no warning


def test_default_bands_partition_up_to_the_top():
    """The default bands are contiguous and the last runs to +inf (clipped to Nyquist)."""
    labels = [b[0] for b in DEFAULT_BANDS]
    assert labels == ["<1 kHz", "1-10 kHz", "10 kHz-Nyquist"]
    assert DEFAULT_BANDS[0][1] == 0.0
    assert DEFAULT_BANDS[-1][2] == np.inf
    for (_, _, hi_prev), (_, lo_next, _) in zip(DEFAULT_BANDS, DEFAULT_BANDS[1:]):
        assert hi_prev == lo_next                    # no gaps between adjacent bands
