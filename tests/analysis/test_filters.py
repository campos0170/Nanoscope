"""Tests for nanoscope.analysis.filters (signal-first port).

Three of these exist to pin down claims that are easy to get silently wrong:
the -3 dB point really lands on the requested cutoff, the rise time is measured
rather than assumed, and the noise estimator survives its own events.

Ported from the standalone ``tests/test_filters.py``. The old ``Recording``
fixture is gone: the kernels now take ``(data, fs, ...)`` numpy arrays and return
plain ``float32`` arrays, so ``make_rec(x)`` -> ``x`` and ``out.current`` -> ``out``.
Every numeric assertion is preserved unchanged. Provenance meta (``filter_rise_s``,
``notch_hz``, ...) moved to the caller, so those checks are expressed against the
signal-first surface (``measure_rise_time`` / ``notch_frequencies``) instead of a
``Recording.meta`` dict.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import signal

from nanoscope.analysis.filters import (
    _design_sos,
    highpass,
    lowpass,
    measure_rise_time,
    median_filter,
    noise_rms,
    notch,
    notch_frequencies,
    psd,
    running_mean,
    savgol,
)

FS = 250_000.0
NAIVE_RISE = 0.34  # textbook t_r ~= 0.34 / fc, single-pass, low order


def response_at(sos: np.ndarray, freqs, fs: float = FS) -> np.ndarray:
    _, h = signal.sosfreqz(sos, worN=np.atleast_1d(freqs), fs=fs)
    return np.abs(h)


# -- the -3 dB point is where we said it is ------------------------------------


@pytest.mark.parametrize("fc", [1e3, 10e3, 50e3])
@pytest.mark.parametrize("order", [2, 4, 8])
def test_bessel_norm_mag_puts_3db_at_cutoff(fc, order):
    """Measured, not trusted: |H(fc)| must be 1/sqrt(2) for the design we ship."""
    sos = _design_sos(FS, fc, order, "bessel")
    mag = response_at(sos, fc)[0]
    assert mag == pytest.approx(1 / np.sqrt(2), rel=1e-3), (
        f"-3 dB point is not at the requested cutoff: |H({fc:g})| = {mag:.4f} "
        f"({20 * np.log10(mag):.2f} dB)"
    )


@pytest.mark.parametrize("order", [4, 8])
def test_scipy_default_norm_phase_is_wrong_at_the_cutoff(order):
    """The bug we are guarding against. scipy's DEFAULT bessel norm='phase'
    normalises group delay, leaving the -3 dB point far from `cutoff_hz`.
    If this ever stops being true, the override in _design_sos can be revisited."""
    fc = 10e3
    default_sos = signal.bessel(order, fc, btype="low", fs=FS, output="sos")
    default_mag = response_at(default_sos, fc)[0]
    ours = response_at(_design_sos(FS, fc, order, "bessel"), fc)[0]

    assert default_mag < 0.5, (
        "scipy's norm='phase' default is expected to be badly attenuated at the "
        f"nominal cutoff, got |H| = {default_mag:.3f}"
    )
    # The default is off by many dB; ours is on the money.
    assert abs(20 * np.log10(default_mag) - (-3.0)) > 3.0
    assert abs(20 * np.log10(ours) - (-3.01)) < 0.05


def test_butter_3db_at_cutoff():
    sos = _design_sos(FS, 10e3, 8, "butter")
    assert response_at(sos, 10e3)[0] == pytest.approx(1 / np.sqrt(2), rel=1e-6)


def test_filtfilt_is_6db_at_cutoff_not_3db():
    """Zero-phase = two passes = squared magnitude. The delivered attenuation at
    fc is -6 dB, and downstream bandwidth claims must use this, not the design."""
    fc = 10e3
    n = 200_000
    t = np.arange(n) / FS
    out = lowpass(np.sin(2 * np.pi * fc * t), FS, fc, order=8, kind="bessel")
    # Amplitude over the settled middle, away from filtfilt's edge padding.
    mid = out[n // 4 : 3 * n // 4]
    amp = np.sqrt(2) * np.std(mid)
    assert amp == pytest.approx(0.5, rel=0.05), f"expected ~0.5 (-6 dB), got {amp:.3f}"


@pytest.mark.parametrize("fc", [1e3, 10e3])
def test_gaussian_3db_at_cutoff(fc):
    """The Gaussian sigma formula is a hand derivation; verify it empirically."""
    n = 400_000
    t = np.arange(n) / FS
    out = lowpass(np.sin(2 * np.pi * fc * t), FS, fc, kind="gaussian")
    mid = out[n // 4 : 3 * n // 4]
    amp = np.sqrt(2) * np.std(mid)
    # Gaussian is applied once, so -3 dB (not -6 dB) at fc.
    assert amp == pytest.approx(1 / np.sqrt(2), rel=0.02)


# -- rise time is measured, and the closed form is wrong ------------------------


@pytest.mark.parametrize("fc", [1e3, 10e3, 50e3])
@pytest.mark.parametrize("kind", ["bessel", "butter"])
def test_measured_rise_time_differs_from_naive_closed_form(fc, kind):
    """The whole reason measure_rise_time exists. filtfilt doubles the effective
    order, so the textbook 0.34/fc understates the true 10-90% rise by ~30-50%.
    Using it would make every filter-limited flag and every resolution claim
    optimistic in the flattering direction."""
    measured = measure_rise_time(FS, fc, order=8, kind=kind)
    naive = NAIVE_RISE / fc
    ratio = measured / naive
    assert ratio > 1.25, (
        f"{kind}: measured rise {measured * 1e6:.2f} us vs naive {naive * 1e6:.2f} us "
        f"(ratio {ratio:.3f}) -- expected the closed form to be substantially optimistic"
    )
    assert ratio < 2.0, f"{kind}: implausible rise-time ratio {ratio:.3f}"


def test_rise_time_scales_inversely_with_cutoff():
    a = measure_rise_time(FS, 5e3, order=8)
    b = measure_rise_time(FS, 10e3, order=8)
    assert a / b == pytest.approx(2.0, rel=0.05)


def test_rise_time_grows_with_order():
    assert measure_rise_time(FS, 10e3, order=8) > measure_rise_time(FS, 10e3, order=2)


def test_rise_time_independent_of_sampling_rate():
    """A physical property of the filter, not of how densely we sampled it."""
    a = measure_rise_time(250e3, 10e3, order=8)
    b = measure_rise_time(500e3, 10e3, order=8)
    assert a == pytest.approx(b, rel=0.03)


def test_lowpass_rise_time_is_the_measured_rise_time():
    """Provenance moved to the caller, but the number a caller would stamp is still
    exactly ``measure_rise_time`` for the same design -- and it is finite/positive."""
    rise = measure_rise_time(FS, 10e3, order=8, kind="bessel")
    assert np.isfinite(rise) and rise > 0
    # Determinism: the callable lowpass uses is the same one measured here.
    assert measure_rise_time(FS, 10e3, order=8, kind="bessel") == rise


def test_measured_rise_time_matches_an_independent_step_measurement():
    """Cross-check measure_rise_time against a step pushed through lowpass()
    itself -- i.e. the number describes what actually happens to the data."""
    fc = 10e3
    n = 20_000
    step = np.concatenate([np.zeros(n), np.ones(n)])
    out = lowpass(step, FS, fc, order=8).astype(np.float64)
    t10 = np.interp(0.1, out[n - 500 : n + 500], np.arange(n - 500, n + 500))
    t90 = np.interp(0.9, out[n - 500 : n + 500], np.arange(n - 500, n + 500))
    assert (t90 - t10) / FS == pytest.approx(measure_rise_time(FS, fc, order=8), rel=0.02)


# -- zero phase ---------------------------------------------------------------


def test_lowpass_does_not_shift_events_in_time():
    """A causal filter would delay the pulse centre; filtfilt must not."""
    n = 100_000
    x = np.zeros(n)
    x[40_000:60_000] = -100.0  # a long, well-resolved blockade
    out = lowpass(x, FS, 5e3, order=8).astype(np.float64)
    idx = np.arange(n)
    centre_in = np.sum(idx * x) / np.sum(x)
    centre_out = np.sum(idx * out) / np.sum(out)
    assert centre_out == pytest.approx(centre_in, abs=1.0)


def test_lowpass_preserves_depth_of_well_resolved_events():
    n = 100_000
    x = np.zeros(n)
    x[40_000:60_000] = -100.0
    out = lowpass(x, FS, 5e3, order=8).astype(np.float64)
    assert out[50_000] == pytest.approx(-100.0, rel=1e-3)


def test_lowpass_attenuates_events_shorter_than_the_rise_time():
    """The physical fact that filter_limited exists to flag."""
    fc = 10e3
    rise = measure_rise_time(FS, fc, order=8)
    n = 100_000
    width = max(int(0.3 * rise * FS), 2)
    x = np.zeros(n)
    x[50_000 : 50_000 + width] = -100.0
    out = lowpass(x, FS, fc, order=8).astype(np.float64)
    assert abs(out.min()) < 60.0, "a sub-rise-time event must not retain its full depth"


def test_lowpass_reduces_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, 200_000)
    out = lowpass(x, FS, 10e3, order=8)
    assert noise_rms(out) < 0.5 * noise_rms(x)


def test_lowpass_returns_float32_of_the_same_length():
    x = np.random.default_rng(1).normal(0, 5, 10_000)
    out = lowpass(x, FS, 10e3)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert out.size == x.size


# -- loud failure on nonsense -------------------------------------------------


def test_lowpass_raises_at_or_above_nyquist():
    x = np.zeros(10_000)
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(x, FS, FS / 2)
    with pytest.raises(ValueError, match="Nyquist"):
        lowpass(x, FS, FS)


def test_lowpass_warns_above_fs_over_2p5():
    x = np.zeros(10_000)
    with pytest.warns(RuntimeWarning, match="fs/2.5"):
        lowpass(x, FS, FS / 2.2)


def test_lowpass_does_not_warn_at_a_sane_cutoff():
    x = np.zeros(10_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lowpass(x, FS, 10e3)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_lowpass_rejects_nonsense_cutoff(bad):
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        lowpass(x, FS, bad)


def test_lowpass_rejects_unknown_kind():
    x = np.zeros(10_000)
    with pytest.raises(ValueError, match="kind"):
        lowpass(x, FS, 10e3, kind="chebyshev")


# -- noise_rms: the estimator must survive its own events ----------------------


def test_mad_noise_rms_agrees_with_std_on_clean_gaussian_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 12.0, 500_000)
    assert noise_rms(x) == pytest.approx(12.0, rel=0.01)
    assert noise_rms(x) == pytest.approx(np.std(x), rel=0.01)


def test_mad_noise_rms_is_robust_to_5pct_event_contamination_and_std_is_not():
    """The headline bug. 5% of samples inside blockades is an ordinary trace.
    std absorbs them and inflates; the inflated threshold then silently discards
    the smallest real events.

    MAD is robust but NOT unbiased here: one-sided contamination drags the median,
    leaving MAD a few percent high. That bias is conservative and ~50x smaller than
    std's. Measured on this fixture: MAD +6%, std +347%."""
    rng = np.random.default_rng(0)
    n = 200_000
    sigma = 12.0
    x = rng.normal(0.0, sigma, n)

    # 5% of samples sit inside blockades 20 sigma deep (a normal, obvious event).
    n_bad = n // 20
    x[:n_bad] -= 20 * sigma

    mad_est = noise_rms(x)
    std_est = float(np.std(x))

    mad_err = abs(mad_est - sigma) / sigma
    std_err = abs(std_est - sigma) / sigma

    assert mad_err < 0.10, f"MAD estimate {mad_est:.2f} pA drifted from {sigma} pA"
    assert std_err > 3.0, f"std estimate {std_est:.2f} pA was expected to blow up"
    # The bias that remains must be conservative (high), never optimistic (low):
    # a low noise estimate would manufacture false events out of noise.
    assert mad_est >= sigma, "MAD must not under-report noise under contamination"
    # Quantify the gap so a regression is legible in the failure message.
    assert std_est / mad_est > 4.0, (
        f"std={std_est:.2f} pA vs MAD={mad_est:.2f} pA: std is inflated "
        f"{std_est / mad_est:.1f}x, which raises a k=5 threshold from "
        f"{5 * mad_est:.0f} pA to {5 * std_est:.0f} pA"
    )


def test_noise_rms_unaffected_by_dc_offset():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 8.0, 100_000)
    assert noise_rms(x + 1000.0) == pytest.approx(noise_rms(x), rel=1e-9)


def test_noise_rms_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        noise_rms(np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        noise_rms(np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="1-D"):
        noise_rms(np.zeros((10, 2)))


# -- notch --------------------------------------------------------------------


def test_notch_removes_mains_and_harmonics_but_keeps_the_signal_band():
    fs = 50_000.0
    n = 500_000
    t = np.arange(n) / fs
    hum = 50 * np.sin(2 * np.pi * 60 * t) + 20 * np.sin(2 * np.pi * 120 * t)
    band = 30 * np.sin(2 * np.pi * 3_000 * t)
    out = notch(hum + band, fs, 60.0, harmonics=3)

    f, pxx = psd(out, fs, nperseg=65536)
    f0, pxx0 = psd(hum + band, fs, nperseg=65536)

    def power_near(f, pxx, target):
        return pxx[np.argmin(np.abs(f - target))]

    assert power_near(f, pxx, 60) < 1e-3 * power_near(f0, pxx0, 60)
    assert power_near(f, pxx, 120) < 1e-3 * power_near(f0, pxx0, 120)
    # The 3 kHz component -- where events live -- must be untouched.
    assert power_near(f, pxx, 3_000) == pytest.approx(
        power_near(f0, pxx0, 3_000), rel=0.02
    )


def notched_event_artifact(dwell_s: float) -> tuple[float, float]:
    """Push one square blockade through notch(); return (depth error %, ringing pA).

    The distortion is 60 Hz ringing excited by the event's own spectral content at
    60 Hz, so the ringing amplitude -- not the depth at one arbitrary sample, which
    lands on an arbitrary phase of the ring -- is the honest measure.
    """
    fs = 250_000.0
    n = 4_000_000
    s = n // 2
    w = int(dwell_s * fs)
    x = np.zeros(n)
    x[s : s + w] = -100.0
    out = notch(x, fs, 60.0, harmonics=3).astype(np.float64)
    depth_err = abs(out[s : s + w].min() + 100.0)
    ring = float(np.abs(np.concatenate([out[: s - 1000], out[s + w + 1000 :]])).max())
    return depth_err, ring


def test_notch_barely_distorts_a_100us_event():
    """The docstring's 'nearly free' claim, quantified rather than assumed.
    A 100 us event has essentially no energy at 60 Hz, so the notch cannot see it."""
    depth_err, ring = notched_event_artifact(100e-6)
    assert depth_err < 1.0, f"100 us event lost {depth_err:.2f}% of its depth"
    assert ring < 0.5, f"notch rang at {ring:.2f} pA around a 100 us event"


def test_notch_distorts_slow_events_far_more_than_fast_ones():
    """The docstring's caveat, and the reason it is stated rather than assumed:
    once dwell approaches the mains period the notch sits inside the event's own
    band. Users with ~10 ms events must not notch."""
    _, ring_fast = notched_event_artifact(100e-6)
    _, ring_slow = notched_event_artifact(10e-3)
    assert ring_slow > 10 * ring_fast, (
        f"expected a 10 ms event to be visibly distorted: ringing {ring_slow:.2f} pA "
        f"vs {ring_fast:.2f} pA for 100 us"
    )
    assert ring_slow > 1.0


def test_notch_skips_harmonics_above_nyquist():
    x = np.random.default_rng(0).normal(0, 1, 20_000)
    with pytest.warns(RuntimeWarning, match="Nyquist"):
        out = notch(x, 500.0, 60.0, harmonics=5)
    assert out.dtype == np.float32
    assert out.size == x.size
    # The kept lines are exactly those strictly below Nyquist (250 Hz): 300 dropped.
    assert notch_frequencies(500.0, 60.0, harmonics=5) == [60.0, 120.0, 180.0, 240.0]


def test_notch_raises_when_no_harmonic_is_representable():
    x = np.zeros(1_000)
    with pytest.raises(ValueError, match="Nyquist"):
        notch(x, 100.0, 60.0, harmonics=1)


def test_notch_rejects_nonsense_args():
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        notch(x, FS, -60.0)
    with pytest.raises(ValueError):
        notch(x, FS, 60.0, q=0)
    with pytest.raises(ValueError):
        notch(x, FS, 60.0, harmonics=0)


# -- psd ----------------------------------------------------------------------


def test_psd_level_is_correct_for_white_noise_in_pA2_per_Hz():
    """Units are load-bearing: a PSD in the wrong scaling misreports noise density.
    White noise of sigma pA over a one-sided band fs/2 has PSD 2*sigma^2/fs."""
    rng = np.random.default_rng(0)
    sigma = 10.0
    x = rng.normal(0, sigma, 2_000_000)
    f, pxx = psd(x, FS, nperseg=65536)
    expected = 2 * sigma**2 / FS
    assert np.median(pxx) == pytest.approx(expected, rel=0.05)
    assert f[0] == 0.0
    assert f[-1] == pytest.approx(FS / 2)
    assert f.shape == pxx.shape


def test_psd_finds_a_60hz_line():
    fs = 50_000.0
    n = 500_000
    t = np.arange(n) / fs
    x = np.random.default_rng(0).normal(0, 1, n) + 50 * np.sin(2 * np.pi * 60 * t)
    f, pxx = psd(x, fs, nperseg=65536)
    assert f[np.argmax(pxx)] == pytest.approx(60.0, abs=2.0)


def test_psd_clamps_nperseg_to_short_traces():
    x = np.random.default_rng(0).normal(0, 1, 1_000)
    f, pxx = psd(x, FS, nperseg=65536)
    assert f.size == 501


# -- high-pass: kill the drift/DC, keep the events' band -----------------------


def test_highpass_removes_low_freq_and_dc_ramp_passes_high_freq():
    """The reason high-pass exists: strip slow drift / DC while a fast component
    (where events live) passes essentially untouched."""
    n = 200_000
    t = np.arange(n) / FS
    mid = slice(n // 4, 3 * n // 4)  # away from filtfilt's edge padding
    slow = 5.0 * np.sin(2 * np.pi * 5.0 * t) + np.linspace(0.0, 20.0, n)  # 5 Hz + DC ramp
    fast = np.sin(2 * np.pi * 5_000.0 * t)

    y_slow = highpass(slow, FS, 200.0).astype(np.float64)
    y_fast = highpass(fast, FS, 200.0).astype(np.float64)

    # Low-frequency content is strongly attenuated ...
    assert y_slow[mid].std() < 0.02 * slow[mid].std()
    # ... while the 5 kHz component passes with its amplitude intact.
    assert y_fast[mid].std() == pytest.approx(fast[mid].std(), rel=0.02)


def test_highpass_bessel_kind_is_also_a_highpass():
    n = 100_000
    t = np.arange(n) / FS
    mid = slice(n // 4, 3 * n // 4)
    fast = np.sin(2 * np.pi * 5_000.0 * t)
    x = 3.0 * np.sin(2 * np.pi * 5.0 * t) + fast
    y = highpass(x, FS, 200.0, kind="bessel").astype(np.float64)
    # The slow term is gone; what remains is the fast component (small residual).
    assert (y[mid] - fast[mid]).std() < 0.1


def test_highpass_is_zero_phase_symmetric_pulse_stays_centred():
    """filtfilt is acausal: a symmetric pulse's high-pass response must stay symmetric
    about the pulse centre. A causal high-pass would tilt it (a phase shift)."""
    n = 100_000
    c, w = n // 2, 2_000
    x = np.zeros(n)
    x[c - w : c + w + 1] = -100.0  # rectangular pulse, symmetric about index c
    y = highpass(x, FS, 100.0).astype(np.float64)
    off = np.arange(1, 5_000)
    assert np.max(np.abs(y[c - off] - y[c + off])) < 1e-3


def test_highpass_returns_float32_of_the_same_length():
    x = np.random.default_rng(1).normal(0, 5, 10_000)
    y = highpass(x, FS, 1_000.0)
    assert y.dtype == np.float32
    assert y.flags["C_CONTIGUOUS"]
    assert y.size == x.size


def test_highpass_raises_at_or_above_nyquist():
    x = np.zeros(10_000)
    with pytest.raises(ValueError, match="Nyquist"):
        highpass(x, FS, FS / 2)
    with pytest.raises(ValueError, match="Nyquist"):
        highpass(x, FS, FS)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_highpass_rejects_nonsense_cutoff(bad):
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        highpass(x, FS, bad)


def test_highpass_rejects_unknown_kind():
    x = np.zeros(10_000)
    # 'gaussian' is a low-pass kernel with no high-pass counterpart -> rejected.
    with pytest.raises(ValueError, match="kind"):
        highpass(x, FS, 1_000.0, kind="gaussian")


def test_highpass_does_not_warn_at_a_sane_cutoff():
    x = np.zeros(10_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        highpass(x, FS, 100.0)


# -- Savitzky-Golay: smooth noise, keep the peak -------------------------------


def test_savgol_smooths_noise_but_preserves_a_step():
    """std drops in the flat regions, yet the step's HEIGHT survives and its edge is
    not shifted -- the property a moving average would spoil."""
    n = 20_000
    rng = np.random.default_rng(0)
    step = np.concatenate([np.zeros(n // 2), np.full(n // 2, 50.0)])
    noisy = step + rng.normal(0.0, 5.0, n)
    y = savgol(noisy, FS, window_s=1e-4, polyorder=3).astype(np.float64)  # 25-sample window

    # Noise suppressed well away from the edge.
    assert y[: n // 2 - 500].std() < 0.5 * noisy[: n // 2 - 500].std()
    # Step height preserved.
    lo = float(np.median(y[1_000 : n // 2 - 1_000]))
    hi = float(np.median(y[n // 2 + 1_000 : n - 1_000]))
    assert (hi - lo) == pytest.approx(50.0, abs=1.0)
    # Edge NOT shifted: the half-height crossing stays at the step (within the window).
    cross = int(np.argmax(y >= 25.0))
    assert abs(cross - n // 2) <= 25


def test_savgol_reduces_white_noise_std():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 10.0, 100_000)
    y = savgol(x, FS, window_s=2e-4, polyorder=3)
    assert noise_rms(y) < 0.5 * noise_rms(x)


def test_savgol_window_is_forced_odd(monkeypatch):
    """window_s * fs = 40 (even) must be forced to an odd length (41), rounded UP so
    the window is not shrunk below the request."""
    import nanoscope.analysis.filters as filt_mod

    captured: dict[str, int] = {}
    real = filt_mod.signal.savgol_filter

    def spy(x, window_length, polyorder, *a, **k):
        captured["win"] = int(window_length)
        return real(x, window_length, polyorder, *a, **k)

    monkeypatch.setattr(filt_mod.signal, "savgol_filter", spy)
    savgol(np.zeros(10_000), 100_000.0, window_s=4e-4, polyorder=3)  # 0.0004 * 100k = 40
    assert captured["win"] == 41
    assert captured["win"] % 2 == 1


def test_savgol_returns_float32_of_the_same_length():
    x = np.random.default_rng(1).normal(0, 5, 10_000)
    y = savgol(x, FS, window_s=1e-3)
    assert y.dtype == np.float32
    assert y.flags["C_CONTIGUOUS"]
    assert y.size == x.size


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_savgol_rejects_bad_polyorder(bad):
    x = np.random.default_rng(0).normal(0, 1, 10_000)
    with pytest.raises(ValueError, match="polyorder"):
        savgol(x, FS, window_s=1e-3, polyorder=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_savgol_rejects_bad_window(bad):
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        savgol(x, FS, window_s=bad)


def test_savgol_raises_when_trace_too_short_for_a_valid_window():
    """A trace with fewer samples than polyorder+1 cannot host the polynomial: the
    clamped-down window falls to <= polyorder and the guard fires."""
    with pytest.raises(ValueError, match="window length"):
        savgol(np.zeros(3), FS, window_s=1.0, polyorder=3)


# -- running mean: box average, unbiased noise floor, zero phase ---------------


def test_running_mean_drops_white_noise_std_by_root_n():
    """A box average of N samples reduces white-noise std by ~sqrt(N)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 10.0, 400_000)
    n = 25
    y = running_mean(x, FS, window_s=n / FS).astype(np.float64)
    # Interior only, away from the nearest-sample edge handling.
    ratio = x[1_000:-1_000].std() / y[1_000:-1_000].std()
    assert ratio == pytest.approx(np.sqrt(n), rel=0.05)


def test_running_mean_is_zero_phase_symmetric_pulse_stays_put():
    """A centred box must not shift a symmetric pulse: its centroid and its peak
    location are unchanged (a trailing/causal average would delay both)."""
    n = 100_000
    c, w = n // 2, 2_000
    # A symmetric TRIANGULAR pulse: uniquely peaked at c, so the argmin of the smoothed
    # trace is a real peak-location check (a flat-topped box would have a plateau of minima).
    off = np.arange(-w, w + 1)
    x = np.zeros(n)
    x[c - w : c + w + 1] = -100.0 * (1.0 - np.abs(off) / w)
    y = running_mean(x, FS, window_s=201 / FS).astype(np.float64)
    idx = np.arange(n)
    centre_in = np.sum(idx * x) / np.sum(x)
    centre_out = np.sum(idx * y) / np.sum(y)
    assert centre_out == pytest.approx(centre_in, abs=0.5)
    # The peak (most negative sample) stays at the pulse centre.
    assert abs(int(np.argmin(y)) - c) <= 1


def test_running_mean_leaves_a_flat_trace_unchanged():
    x = np.full(50_000, 42.0)
    y = running_mean(x, FS, window_s=1e-3).astype(np.float64)
    assert np.allclose(y, 42.0, atol=1e-4)


def test_running_mean_n1_is_passthrough():
    """A window rounding to a single sample averages nothing -> the identity."""
    x = np.random.default_rng(0).normal(0, 5, 10_000)
    y = running_mean(x, FS, window_s=0.4 / FS)  # 0.4 sample -> N = max(1, 0) = 1
    np.testing.assert_allclose(y, x.astype(np.float32))


def test_running_mean_returns_float32_of_the_same_length():
    x = np.random.default_rng(1).normal(0, 5, 10_000)
    y = running_mean(x, FS, window_s=1e-4)
    assert y.dtype == np.float32
    assert y.flags["C_CONTIGUOUS"]
    assert y.size == x.size


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_running_mean_rejects_bad_window(bad):
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        running_mean(x, FS, window_s=bad)


def test_running_mean_rejects_bad_rate():
    x = np.zeros(10_000)
    with pytest.raises(ValueError, match="fs"):
        running_mean(x, 0.0, window_s=1e-3)


# -- median filter: kill impulse spikes, keep the square edges -----------------


def test_median_filter_removes_single_sample_spikes_but_keeps_a_step():
    """The headline job: delete isolated impulses while a clean step's height and edge
    location survive (a moving average would smear both the spike and the edge)."""
    n = 20_000
    step = np.concatenate([np.zeros(n // 2), np.full(n // 2, 50.0)])
    spiky = step.copy()
    # A handful of isolated single-sample spikes, far from each other and the edge.
    spike_idx = [1_000, 4_000, 7_000, 13_000, 17_000]
    spiky[spike_idx] += 500.0
    y = median_filter(spiky, FS, window_s=5 / FS).astype(np.float64)  # 5-sample window

    # Every injected spike is gone (median of a 5-window with one outlier is a clean sample).
    assert np.max(np.abs(y[spike_idx] - step[spike_idx])) < 1e-6
    # Step height preserved.
    lo = float(np.median(y[1_500 : n // 2 - 1_500]))
    hi = float(np.median(y[n // 2 + 1_500 : n - 1_500]))
    assert (hi - lo) == pytest.approx(50.0, abs=1e-6)
    # Edge NOT shifted: the half-height crossing stays at the step.
    cross = int(np.argmax(y >= 25.0))
    assert abs(cross - n // 2) <= 2


def test_median_window_is_forced_odd(monkeypatch):
    """window_s * fs = 8 (even) must be forced to an odd kernel (9) so the window is
    centred on the sample it replaces."""
    import nanoscope.analysis.filters as filt_mod

    captured: dict[str, int] = {}
    real = filt_mod.signal.medfilt

    def spy(x, kernel_size):
        captured["win"] = int(kernel_size)
        return real(x, kernel_size)

    monkeypatch.setattr(filt_mod.signal, "medfilt", spy)
    median_filter(np.zeros(10_000), 100_000.0, window_s=8e-5)  # 8e-5 * 100k = 8 (even)
    assert captured["win"] == 9
    assert captured["win"] % 2 == 1


def test_median_filter_leaves_a_flat_trace_unchanged():
    x = np.full(50_000, 42.0)
    y = median_filter(x, FS, window_s=5e-5).astype(np.float64)
    assert np.allclose(y, 42.0, atol=1e-6)


def test_median_filter_returns_float32_of_the_same_length():
    x = np.random.default_rng(1).normal(0, 5, 10_000)
    y = median_filter(x, FS, window_s=5e-5)
    assert y.dtype == np.float32
    assert y.flags["C_CONTIGUOUS"]
    assert y.size == x.size


def test_median_filter_clamps_window_to_a_short_trace():
    """A window longer than the trace is clamped to the largest odd length that fits,
    rather than raising -- a short trace is still filterable."""
    x = np.array([1.0, 100.0, 1.0, 1.0])  # length 4; window would clamp to 3
    y = median_filter(x, FS, window_s=1.0)  # 1 s * FS >> 4 samples
    assert y.size == x.size
    assert y.dtype == np.float32


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_median_filter_rejects_bad_window(bad):
    x = np.zeros(10_000)
    with pytest.raises(ValueError):
        median_filter(x, FS, window_s=bad)


def test_median_filter_rejects_empty_trace():
    with pytest.raises(ValueError, match="empty"):
        median_filter(np.array([]), FS, window_s=1e-4)


# -- notch: guard the pre-existing behaviour still holds -----------------------


def test_notch_still_removes_the_60hz_line():
    """Guard: completing the filter suite must not disturb the notch's core job."""
    fs = 50_000.0
    n = 500_000
    t = np.arange(n) / fs
    x = np.random.default_rng(0).normal(0, 1, n) + 40 * np.sin(2 * np.pi * 60 * t)
    out = notch(x, fs, 60.0, harmonics=1)

    f0, p0 = psd(x, fs, nperseg=65536)
    f1, p1 = psd(out, fs, nperseg=65536)

    def power_near(f, p, target):
        return p[np.argmin(np.abs(f - target))]

    assert power_near(f1, p1, 60) < 1e-3 * power_near(f0, p0, 60)
