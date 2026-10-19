"""The Response tab's phase, group-delay and step views, held to scipy.

Applied = what NanoScope's zero-phase kernels do; reference = the same IIR design run
once, causally. Every comparison is against an independent scipy design to ≤ 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage, signal

from nanoscope.analysis import filter_views as fv
from nanoscope.analysis import filters

FS = 50_000.0
LP = {"stage": "lowpass", "cutoff_hz": 10_000.0, "order": 8, "kind": "bessel"}
HP = {"stage": "highpass", "cutoff_hz": 100.0, "order": 4, "kind": "butter"}
NOTCH = {"stage": "notch", "freq_hz": 60.0, "q": 30.0, "harmonics": 3, "freqs": None}


def _lp_sos():
    return signal.bessel(8, 10_000.0, fs=FS, output="sos", norm="mag")


def _hp_sos():
    return signal.butter(4, 100.0, btype="high", fs=FS, output="sos")


def _notch_sos():
    return np.vstack([np.hstack(signal.iirnotch(f, 30.0, fs=FS)) for f in (60, 120, 180)])


def test_the_magnitude_view_is_the_squared_sosfreqz():
    f, mag = filters.combined_response([HP, LP], FS, n=2048)
    _, h_lp = signal.sosfreqz(_lp_sos(), worN=f, fs=FS)
    _, h_hp = signal.sosfreqz(_hp_sos(), worN=f, fs=FS)
    np.testing.assert_allclose(mag, np.abs(h_lp) ** 2 * np.abs(h_hp) ** 2, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("spec,sos", [(LP, _lp_sos), (HP, _hp_sos), (NOTCH, _notch_sos)])
def test_the_reference_phase_is_scipys_single_pass(spec, sos):
    ph = fv.chain_phase([spec], FS)
    _, h = signal.sosfreqz(sos(), worN=ph.f_hz, fs=FS)
    assert ph.has_reference
    np.testing.assert_allclose(ph.reference_phase_rad, np.unwrap(np.angle(h)), atol=1e-9)


def test_the_reference_group_delay_matches_scipy_and_the_phase_slope():
    ph = fv.chain_phase([LP], FS)
    sos = _lp_sos()
    with pytest.warns(UserWarning):   # scipy: "denominator extremely small" at Nyquist
        gd = sum(signal.group_delay((s[:3], s[3:]), w=ph.f_hz, fs=FS)[1] for s in sos) / FS
    finite = np.isfinite(ph.reference_group_delay_s)
    # The Bessel's zeros sit at Nyquist, ON the unit circle: undefined there, NaN not a number.
    assert np.flatnonzero(~finite).tolist() == [ph.f_hz.size - 1]
    np.testing.assert_allclose(ph.reference_group_delay_s[finite], gd[finite], rtol=1e-9)
    hp = fv.chain_phase([HP], FS)
    w = 2 * np.pi * hp.f_hz
    slope = -np.gradient(hp.reference_phase_rad, w)
    band = hp.f_hz > 300.0
    np.testing.assert_allclose(hp.reference_group_delay_s[band], slope[band], rtol=2e-4)
    # Flat in band: the Bessel's defining property.
    lo = ph.f_hz < 2_000.0
    assert np.ptp(ph.reference_group_delay_s[lo]) < 0.03 * ph.reference_group_delay_s[0]


def test_the_group_delay_is_deterministic():
    """scipy.signal.group_delay is not (in this environment); the closed form must be."""
    runs = [fv.chain_phase([HP, NOTCH], FS).reference_group_delay_s for _ in range(20)]
    for r in runs[1:]:
        np.testing.assert_allclose(r, runs[0], rtol=1e-12, atol=0, equal_nan=True)


def test_zero_phase_iir_stages_apply_no_phase_and_no_delay():
    ph = fv.chain_phase([NOTCH, HP, LP], FS)
    assert np.all(ph.applied_phase_rad == 0.0)
    assert np.all(ph.applied_group_delay_s == 0.0)


def _impulse_response_phase(apply, f, n=4096):
    """The phase of a kernel's response, from its own impulse response (centred at n//2)."""
    x = np.zeros(n)
    x[n // 2] = 1.0
    h = apply(x)
    k = np.arange(n) - n // 2
    w = 2 * np.pi * f / FS
    return np.exp(-1j * np.outer(w, k)) @ h


@pytest.mark.parametrize("window_s", [9 / FS, 10 / FS])   # an odd and an even box
def test_the_running_mean_phase_is_its_kernels(window_s):
    spec = {"stage": "running_mean", "window_s": window_s}
    ph = fv.chain_phase([spec], FS)
    n = filters.running_mean_samples(FS, window_s)
    H = _impulse_response_phase(lambda x: ndimage.uniform_filter1d(x, n, mode="nearest"),
                                ph.f_hz)
    keep = np.abs(H) > 1e-6
    np.testing.assert_allclose(np.exp(1j * ph.applied_phase_rad[keep]),
                               H[keep] / np.abs(H[keep]), atol=1e-9)
    assert ph.applied_group_delay_s[0] == pytest.approx(0.0 if n % 2 else 0.5 / FS)
    flips = np.count_nonzero(np.abs(np.diff(np.unwrap(ph.applied_phase_rad))) > 1.0)
    assert flips >= 2, "the box's side lobes alternate sign"


def test_the_savgol_phase_is_its_kernels():
    spec = {"stage": "savgol", "window_s": 1e-3, "polyorder": 3}
    ph = fv.chain_phase([spec], FS)
    win = filters.savgol_window_samples(FS, 1e-3, 3)
    H = _impulse_response_phase(lambda x: signal.savgol_filter(x, win, 3), ph.f_hz)
    keep = np.abs(H) > 1e-6
    np.testing.assert_allclose(np.exp(1j * ph.applied_phase_rad[keep]),
                               H[keep] / np.abs(H[keep]), atol=1e-9)


def test_the_applied_step_is_what_the_kernels_apply():
    st = fv.chain_step([LP], FS)
    n = st.t_s.size // 2
    x = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])
    np.testing.assert_allclose(st.applied, signal.sosfiltfilt(_lp_sos(), x)[n:3 * n],
                               atol=1e-12)
    np.testing.assert_allclose(st.applied, filters.lowpass(x, FS, 10_000.0)[n:3 * n],
                               atol=1e-6)    # the kernel returns float32
    for spec, kernel in (
            (NOTCH, lambda x: filters.notch(x, FS, 60.0, q=30.0, harmonics=3)),
            (HP, lambda x: filters.highpass(x, FS, 100.0)),
            ({"stage": "savgol", "window_s": 1e-3, "polyorder": 3},
             lambda x: filters.savgol(x, FS, window_s=1e-3, polyorder=3)),
            ({"stage": "running_mean", "window_s": 5e-4},
             lambda x: filters.running_mean(x, FS, window_s=5e-4))):
        st = fv.chain_step([spec], FS, span_s=0.004)
        n = st.t_s.size // 2
        x = np.concatenate([np.zeros(2 * n), np.ones(2 * n)])
        np.testing.assert_allclose(st.applied, kernel(x)[n:3 * n], atol=2e-6), spec


def test_the_reference_step_is_scipys_dstep():
    st = fv.chain_step([LP], FS)
    n = st.t_s.size // 2
    z, p, k = signal.bessel(8, 10_000.0, fs=FS, output="zpk", norm="mag")
    _t, (y,) = signal.dstep((z, p, k, 1 / FS), n=n)
    m = min(n, y.size)
    np.testing.assert_allclose(st.reference[n:n + m], y.ravel()[:m], atol=1e-9)
    assert np.all(st.reference[:n] == 0.0), "causal: nothing before the step"
    hp2 = {"stage": "highpass", "cutoff_hz": 1_000.0, "order": 2, "kind": "butter"}
    st = fv.chain_step([hp2], FS, span_s=0.004)
    n = st.t_s.size // 2
    b, a = signal.butter(2, 1_000.0, btype="high", fs=FS)
    _t, (y,) = signal.dstep((b, a, 1 / FS), n=n)
    m = min(n, y.size)
    np.testing.assert_allclose(st.reference[n:n + m], y.ravel()[:m], atol=1e-9)


def test_the_step_readout_measures_the_edge():
    st = fv.chain_step([LP], FS)
    rise = filters.measure_rise_time(FS, 10_000.0)
    assert st.applied_rise_s == pytest.approx(rise, rel=0.02)
    assert st.reference_rise_s < st.applied_rise_s, "one pass is faster than two"
    hp = fv.chain_step([HP], FS)
    assert hp.removes_dc and hp.applied_rise_s is None


def test_the_step_span_follows_the_edge_shaping_stages():
    assert fv.step_span_s([LP], FS) == pytest.approx(4 / 10_000.0)
    assert fv.step_span_s([LP, NOTCH], FS) == pytest.approx(4 / 10_000.0)
    assert fv.step_span_s([HP], FS) == pytest.approx(1 / 100.0)
    assert fv.step_span_s([], FS) == pytest.approx(1e-3)
    tiny = {"stage": "highpass", "cutoff_hz": 0.01, "order": 4, "kind": "butter"}
    assert fv.chain_step([tiny], FS).truncated


def test_linear_stage_specs_names_what_has_no_linear_response():
    params = {"notch_enabled": True, "median_enabled": True, "hp_enabled": True,
              "detrend_enabled": True, "normalize_enabled": True, "filter_cutoff_hz": 8_000.0}
    stages, omitted = fv.linear_stage_specs(params)
    assert [s["stage"] for s in stages] == ["notch", "highpass", "lowpass"]
    assert omitted == ["Median (nonlinear rank filter)",
                       "Baseline drift (signal-dependent estimator)",
                       "Normalize (data-dependent affine/nonlinear rescale)"]
    off, _ = fv.linear_stage_specs({}, only="lowpass", lowpass_design_hz=5_000.0)
    assert off == [{"stage": "lowpass", "cutoff_hz": 5_000.0, "order": 8, "kind": "bessel"}]
    assert fv.linear_stage_specs({}, only="median") == ([], ["Median (nonlinear rank filter)"])


def test_an_invalid_stage_raises_like_combined_response():
    bad = {"stage": "lowpass", "cutoff_hz": 30_000.0}
    with pytest.raises(ValueError) as a:
        filters.combined_response([bad], FS)
    with pytest.raises(ValueError) as b:
        fv.chain_phase([bad], FS)
    with pytest.raises(ValueError) as c:
        fv.chain_step([bad], FS)
    assert str(a.value) == str(b.value) == str(c.value)
