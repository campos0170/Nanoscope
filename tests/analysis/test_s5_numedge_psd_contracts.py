"""Sweep-5 numeric-edge regressions in analysis.psd: NE-07 (underflow honesty) and
the NE-03 seam (band_support == what band_rms integrates).

NE-07: a 1e-310-scale live signal squares to below float64's smallest subnormal
inside Welch, so every PSD bin reads exactly 0.0 — finite, so none of band_rms's
guards fired — and noise_summary reported "total_rms 0.0" with zero warnings for a
signal the app's own filters.noise_rms measures at ~2e-312. band_rms's contract:
"'no noise in this band' and 'this band cannot be measured' are different answers,
and only one of them may be printed as a number." welch_psd is the one place both
the samples and the spectrum are in hand, so it now warns there.

NE-03 seam: the band a label restates and the band band_rms integrates must be ONE
arithmetic — band_support — or they drift apart again.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanoscope.analysis import filters as nsfilters
from nanoscope.analysis.psd import band_rms, band_support, noise_summary, welch_psd

FS = 50_000.0


def _subnormal_signal(n: int = 200_000, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1e-310 + rng.normal(0.0, 1.0, n) * 2e-312


# ================================================================================================
# NE-07: the confident zero is refused at the source
# ================================================================================================

def test_welch_warns_when_a_live_signal_underflows_to_an_all_zero_spectrum():
    x = _subnormal_signal()
    assert float(nsfilters.noise_rms(x)) > 0.0        # the app CAN measure this noise
    with pytest.warns(RuntimeWarning, match="below-representable, not zero"):
        freqs, psd = welch_psd(x, FS)
    assert np.isfinite(psd).all() and not psd.any()   # the underflow itself is real


def test_the_warning_names_the_representation_cause_and_a_remedy():
    with pytest.warns(RuntimeWarning) as rec:
        welch_psd(_subnormal_signal(), FS)
    msg = str(rec[0].message)
    assert "square to below float64" in msg
    assert "unmeasurable from this spectrum" in msg
    assert "Rescale" in msg


def test_a_constant_signal_keeps_its_honest_zero_with_no_warning():
    """min == max is the OTHER all-zero spectrum, and it is genuinely zero variance —
    warning there would cry wolf on every flat-line channel."""
    x = np.full(65_536, 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # any warning fails the test
        freqs, psd = welch_psd(x, FS)
    assert not psd.any()
    assert noise_summary(freqs, psd, FS)["total_rms"] == 0.0


def test_an_ordinary_noisy_signal_warns_nothing():
    x = np.random.default_rng(0).normal(0.0, 3.0, 1 << 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        freqs, psd = welch_psd(x, FS)
    assert band_rms(freqs, psd, 0.0, FS / 2) == pytest.approx(3.0, rel=0.05)


def test_a_barely_representable_signal_integrates_without_the_warning():
    """sigma = 1e-150 squares to 1e-300 — representable. The RMS comes back as the
    honest tiny number; the warning is for LOST variance only."""
    x = np.random.default_rng(1).normal(0.0, 1e-150, 1 << 16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        freqs, psd = welch_psd(x, FS)
    assert band_rms(freqs, psd, 0.0, FS / 2) == pytest.approx(1e-150, rel=0.1)


# ================================================================================================
# NE-03 seam: one clip arithmetic for the integral and any label of it
# ================================================================================================

def test_band_support_is_the_clip_band_rms_integrates():
    freqs, psd = welch_psd(np.random.default_rng(0).normal(0, 1, 1 << 16), 1.0)
    nyq = float(freqs.max())
    assert nyq == pytest.approx(0.5)
    assert band_support(freqs, 0.0, 0.5) == (0.0, 0.5)
    assert band_support(freqs, 0.0, 2.0) == (0.0, 0.5)      # clipped down to Nyquist
    assert band_support(freqs, -1.0, 0.3) == (0.0, 0.3)     # clipped up to 0


def test_band_support_names_the_empty_cases_none():
    freqs = np.linspace(0.0, 0.5, 100)
    assert band_support(freqs, 0.0, 0.0) is None            # degenerate
    assert band_support(freqs, 0.4, 0.2) is None            # inverted
    assert band_support(freqs, 1.0, 2.0) is None            # wholly above Nyquist
    assert band_support(np.empty(0), 0.0, 1.0) is None      # no spectrum at all


def test_band_rms_agrees_with_band_support_on_emptiness():
    """band_rms returns its honest 0.0 exactly when band_support says None — the
    empty-band definitions may not drift apart."""
    freqs, psd = welch_psd(np.random.default_rng(0).normal(0, 1, 1 << 14), FS)
    for lo, hi in [(0.0, 0.0), (5.0, 2.0), (30_000.0, 40_000.0), (0.0, 100.0)]:
        rms = band_rms(freqs, psd, lo, hi)
        if band_support(freqs, lo, hi) is None:
            assert rms == 0.0                       # genuinely nothing in the band
        else:
            assert rms > 0.0                        # white noise: every real band has power


def test_band_rms_values_are_unchanged_by_the_refactor():
    """The support extraction is a refactor of band_rms's own clip: the numbers the
    whole app quotes must be bit-identical to the direct integration."""
    from nanoscope.analysis._compat import trapezoid
    freqs, psd = welch_psd(np.random.default_rng(2).normal(0, 2.5, 1 << 16), FS)
    for lo, hi in [(0.0, FS / 2), (100.0, 1000.0), (9.5, 10.5), (0.0, 1e9)]:
        clo, chi = max(lo, 0.0), min(hi, float(freqs.max()))
        sel = (freqs >= clo) & (freqs <= chi)
        expect = float(np.sqrt(trapezoid(psd[sel], freqs[sel])))
        assert band_rms(freqs, psd, lo, hi) == expect
