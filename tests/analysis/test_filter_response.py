"""The frequency-response helper, and the user's filter parameters reaching the kernels.

Two claims are pinned here, because both used to be unverifiable from inside the app:

1. :func:`nanoscope.analysis.filters.response` describes the filter that is ACTUALLY
   APPLIED -- not the textbook design. Every zero-phase IIR stage runs through
   ``sosfiltfilt``, which applies the designed response twice, so the honest curve is
   |H|^2 and the nominal cutoff is a -6 dB point. The tests check that against an
   independently computed ``sosfreqz`` AND against the empirical gain the real kernels
   put on real sinusoids.
2. The order / kind / baseline-shape parameters the panel now exposes are FORWARDED, and
   their defaults reproduce the previously frozen behaviour exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from nanoscope.analysis import filters
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.modules import (
    DETREND_PARAM_KEYS,
    FILTER_PARAM_KEYS,
    detrend_method_kwargs,
    preprocess_for_detection,
)

FS = 250_000.0


# ================================================================================================
# helpers
# ================================================================================================

def _sine_gain(fn, fs: float, f_hz: float, n: int = 200_000) -> float:
    """Empirical amplitude gain the callable ``fn`` puts on a pure tone at ``f_hz``.

    Measured on the middle half of the output so the filter's edge transients never enter
    the estimate. This is the ground truth ``response`` has to reproduce.
    """
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f_hz * t)
    y = np.asarray(fn(x), dtype=np.float64)
    core = slice(n // 4, 3 * n // 4)
    return float(np.ptp(y[core]) / np.ptp(x[core]))


def _trace_with_events(fs: float = FS, n: int = 200_000, seed: int = 0):
    """A -1000 pA open-pore trace with six well-resolved 300 pA blockades."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -1000.0) + rng.normal(0.0, 12.0, n)
    for i in range(6):
        s = 10_000 + i * 25_000
        x[s : s + 500] += 300.0          # 2 ms blockades, far above any rise time
    return x.astype(np.float32)


# ================================================================================================
# response() vs an independent scipy computation
# ================================================================================================

@pytest.mark.parametrize("kind,order", [("bessel", 8), ("butter", 4), ("bessel", 2)])
def test_lowpass_response_matches_independent_sosfreqz(kind, order):
    """The curve is exactly |sosfreqz|^2 for the same design, computed here from scratch."""
    f, mag = filters.response("lowpass", FS, cutoff_hz=10_000.0, order=order, kind=kind)
    if kind == "bessel":
        sos = signal.bessel(order, 10_000.0, btype="low", fs=FS, output="sos", norm="mag")
    else:
        sos = signal.butter(order, 10_000.0, btype="low", fs=FS, output="sos")
    _, h = signal.sosfreqz(sos, worN=f, fs=FS)
    # Not bitwise: scipy's sosfreqz accumulates in a non-deterministic order, so two
    # calls with identical arguments differ in the last few ULPs. 1e-8 relative is
    # still 'the same computation', many orders tighter than any physical claim here.
    assert np.allclose(mag, np.abs(h) ** 2, rtol=1e-8, atol=1e-14)


def test_highpass_response_matches_independent_sosfreqz():
    f, mag = filters.response("highpass", FS, cutoff_hz=100.0, order=4, kind="butter")
    sos = signal.butter(4, 100.0, btype="high", fs=FS, output="sos")
    _, h = signal.sosfreqz(sos, worN=f, fs=FS)
    # Not bitwise: scipy's sosfreqz accumulates in a non-deterministic order, so two
    # calls with identical arguments differ in the last few ULPs. 1e-8 relative is
    # still 'the same computation', many orders tighter than any physical claim here.
    assert np.allclose(mag, np.abs(h) ** 2, rtol=1e-8, atol=1e-14)


def test_zero_phase_flag_is_exactly_the_square_of_the_single_pass_design():
    f1, single = filters.response(
        "lowpass", FS, cutoff_hz=10_000.0, order=8, kind="bessel", zero_phase=False)
    f2, doubled = filters.response(
        "lowpass", FS, cutoff_hz=10_000.0, order=8, kind="bessel", zero_phase=True)
    assert np.allclose(f1, f2)
    assert np.allclose(doubled, single**2)
    # And the single pass really is the -3 dB design the control's label implies.
    assert np.interp(10_000.0, f1, single) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_applied_response_is_6db_at_the_nominal_cutoff_not_3db():
    """THE point of the preview: the number in the box is not the -3 dB corner."""
    f, mag = filters.response("lowpass", FS, cutoff_hz=10_000.0, order=8, kind="bessel")
    db_at_nominal = float(np.interp(10_000.0, f, filters.response_db(mag)))
    assert db_at_nominal == pytest.approx(-6.02, abs=0.1)

    true_corner = filters.response_crossing_hz(f, mag)
    assert true_corner == pytest.approx(7140.0, rel=0.02)     # ~0.714 * fc
    assert true_corner < 10_000.0


# ================================================================================================
# response() vs what the kernels really do to a signal
# ================================================================================================

@pytest.mark.parametrize("f_probe", [500.0, 2_000.0, 7_000.0, 10_000.0, 20_000.0])
def test_lowpass_response_predicts_the_measured_gain(f_probe):
    _, mag = filters.response(
        "lowpass", FS, cutoff_hz=10_000.0, order=8, kind="bessel", f_min_hz=f_probe, n=2)
    predicted = float(mag[0])
    measured = _sine_gain(
        lambda x: filters.lowpass(x, FS, 10_000.0, order=8, kind="bessel"), FS, f_probe)
    assert measured == pytest.approx(predicted, rel=0.02, abs=1e-3)


@pytest.mark.parametrize("f_probe", [500.0, 5_000.0, 15_000.0])
def test_gaussian_response_predicts_the_measured_gain(f_probe):
    """The Gaussian is a single symmetric convolution -- NOT squared."""
    _, mag = filters.response(
        "lowpass", FS, cutoff_hz=10_000.0, kind="gaussian", f_min_hz=f_probe, n=2)
    measured = _sine_gain(
        lambda x: filters.lowpass(x, FS, 10_000.0, kind="gaussian"), FS, f_probe)
    assert measured == pytest.approx(float(mag[0]), rel=0.02, abs=1e-3)


@pytest.mark.parametrize("f_probe", [200.0, 900.0, 3_000.0])
def test_running_mean_response_predicts_the_measured_gain(f_probe):
    _, mag = filters.response("running_mean", FS, window_s=5e-4, f_min_hz=f_probe, n=2)
    measured = _sine_gain(lambda x: filters.running_mean(x, FS, window_s=5e-4), FS, f_probe)
    assert measured == pytest.approx(float(mag[0]), rel=0.03, abs=2e-3)


@pytest.mark.parametrize("f_probe", [200.0, 1_000.0, 3_000.0])
def test_savgol_response_predicts_the_measured_gain(f_probe):
    _, mag = filters.response("savgol", FS, window_s=1e-3, polyorder=3, f_min_hz=f_probe, n=2)
    measured = _sine_gain(
        lambda x: filters.savgol(x, FS, window_s=1e-3, polyorder=3), FS, f_probe)
    assert measured == pytest.approx(float(mag[0]), rel=0.03, abs=2e-3)


def test_notch_response_is_deep_at_the_line_and_flat_in_the_event_band():
    # A Q=30 notch at 60 Hz is only ~2 Hz wide, five decades below Nyquist, so the grid
    # has to be dense enough to land inside it (the panel asks for 4096 points for
    # exactly this reason).
    f, mag = filters.response("notch", FS, freq_hz=60.0, q=30.0, harmonics=3, n=20_000)
    for line in (60.0, 120.0, 180.0):
        assert float(np.interp(line, f, mag)) < 0.05      # >26 dB down at every line
    assert float(np.interp(10_000.0, f, mag)) == pytest.approx(1.0, abs=1e-3)


# ================================================================================================
# response(): the smoothers band-limit harder than the low-pass (the invisible hazard)
# ================================================================================================

def test_default_savgol_is_an_order_of_magnitude_narrower_than_the_default_lowpass():
    """The panel defaults band-limit the trace ~8x harder via Savitzky-Golay than via the
    10 kHz low-pass, and only the low-pass has a rise-time readout. Now it is on a plot."""
    f_lp, m_lp = filters.response("lowpass", FS, cutoff_hz=10_000.0, order=8, kind="bessel")
    f_sg, m_sg = filters.response("savgol", FS, window_s=1e-3, polyorder=3)
    lp_corner = filters.response_crossing_hz(f_lp, m_lp)
    sg_corner = filters.response_crossing_hz(f_sg, m_sg)
    assert sg_corner < lp_corner / 5.0


def test_running_mean_window_that_rounds_to_one_sample_is_a_flat_identity():
    """A 10 us window below 150 kHz rounds to N=1: the stage is enabled and does nothing.
    The response says so (flat 0 dB) instead of drawing an idealised box."""
    _, mag = filters.response("running_mean", 10_000.0, window_s=1e-5)
    assert np.allclose(mag, 1.0)


# ================================================================================================
# response(): the stages that have no transfer function
# ================================================================================================

def test_median_has_no_response_and_says_why():
    with pytest.raises(ValueError, match="non-linear rank filter"):
        filters.response("median", FS, window_s=1e-4)


def test_unknown_stage_rejected():
    with pytest.raises(ValueError, match="stage must be one of"):
        filters.response("wavelet", FS, cutoff_hz=1000.0)


def test_response_rejects_a_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        filters.response("lowpass", FS, cutoff_hz=FS / 2)


def test_response_grid_ends_at_nyquist_and_is_log_spaced():
    f, _ = filters.response("lowpass", FS, cutoff_hz=10_000.0, n=256)
    assert f.size == 256
    assert f[-1] == pytest.approx(FS / 2)
    ratios = f[1:] / f[:-1]
    assert np.allclose(ratios, ratios[0])


def test_response_db_floors_a_true_zero_instead_of_returning_inf():
    db = filters.response_db(np.array([0.0, 1.0, 0.5]))
    assert np.isfinite(db).all()
    assert db[1] == pytest.approx(0.0)
    assert db[2] == pytest.approx(-6.0206, abs=1e-3)


def test_response_crossing_returns_none_when_the_curve_never_crosses():
    f = np.array([1.0, 2.0, 3.0])
    assert filters.response_crossing_hz(f, np.ones(3)) is None


def test_rise_time_probe_samples_is_the_size_measure_rise_time_really_uses():
    assert filters.rise_time_probe_samples(FS, 1.0) == int(np.ceil(40.0 * FS))
    assert filters.rise_time_probe_samples(FS, 1e9) == 512     # floor


# ================================================================================================
# detect_events: filter_order / filter_kind reach the kernel
# ================================================================================================

def test_butterworth_order_2_gives_a_measurably_different_result_than_the_bessel_8_default():
    """The whole point of exposing the design: it has to actually change the numbers."""
    x = _trace_with_events()
    kw = dict(filter_cutoff_hz=3_000.0, min_dwell_s=1e-4, k_in=8.0)
    df_a, prov_a = detect_events(x, FS, **kw)
    df_b, prov_b = detect_events(x, FS, **kw, filter_kind="butter", filter_order=2)

    assert prov_a["filter_kind"] == "bessel" and prov_a["filter_order"] == 8
    assert prov_b["filter_kind"] == "butter" and prov_b["filter_order"] == 2
    # The rise time is measured through whichever filter ran, so it must differ...
    assert prov_a["filter_rise_s"] != pytest.approx(prov_b["filter_rise_s"], rel=1e-6)
    # ...and so must the trace the detector characterised the events on.
    assert prov_a["noise_rms_pA"] != pytest.approx(prov_b["noise_rms_pA"], rel=1e-6)
    # Both still find the six real events, but they measure them through different
    # filters, so the depths are not the same numbers.
    assert len(df_a) == len(df_b) == 6
    assert not np.allclose(df_a["depth_pA"].to_numpy(), df_b["depth_pA"].to_numpy())


def test_default_kind_and_order_reproduce_the_previously_frozen_behaviour():
    """Omitting the new kwargs must be byte-for-byte the old call."""
    x = _trace_with_events()
    df_a, prov_a = detect_events(x, FS, filter_cutoff_hz=5_000.0, min_dwell_s=1e-4)
    df_b, prov_b = detect_events(
        x, FS, filter_cutoff_hz=5_000.0, min_dwell_s=1e-4,
        filter_kind="bessel", filter_order=8)
    # `filter_rise_s` is MEASURED, by pushing a step through scipy's sosfiltfilt, and the
    # answer depends on the alignment numpy happened to give the work arrays: under a full
    # 14-minute suite run this assertion failed at
    #   9.627945888575687e-05 == 9.627945888575778e-05
    # (9 identical significant digits, ~1e-14 relative) while passing 5/5 in isolation. The
    # contract being frozen here is "the default kwargs are the old call", which a tight
    # relative tolerance states exactly; bit-equality on a float that scipy re-derives every
    # call states something the platform does not promise. The numbers that MUST be
    # bit-identical -- the event geometry and every schema column -- are asserted below
    # with np.array_equal, and those did not move in the failing run either.
    assert prov_a["filter_rise_s"] == pytest.approx(prov_b["filter_rise_s"], rel=1e-9)
    assert prov_a["noise_rms_pA"] == pytest.approx(prov_b["noise_rms_pA"], rel=1e-12)
    assert np.array_equal(df_a["depth_pA"].to_numpy(), df_b["depth_pA"].to_numpy())


def test_gaussian_kind_works_and_ignores_the_order():
    x = _trace_with_events()
    df_1, prov_1 = detect_events(
        x, FS, filter_cutoff_hz=3_000.0, min_dwell_s=1e-4,
        filter_kind="gaussian", filter_order=1)
    df_8, prov_8 = detect_events(
        x, FS, filter_cutoff_hz=3_000.0, min_dwell_s=1e-4,
        filter_kind="gaussian", filter_order=8)
    assert prov_1["filter_rise_s"] == prov_8["filter_rise_s"]
    assert np.array_equal(df_1["depth_pA"].to_numpy(), df_8["depth_pA"].to_numpy())
    # A symmetric kernel has no order, so provenance records None rather than a fiction.
    assert prov_1["filter_kind"] == "gaussian" and prov_1["filter_order"] is None
    assert len(df_1) == 6


def test_the_stamped_rise_time_depends_on_the_kind_and_the_order():
    """The resolution floor is a property of the design, not just the cutoff.

    Note it is NOT monotonic in order: `_design_sos` uses ``norm='mag'``, which pins the
    -3 dB point at the cutoff whatever the order, so a Bessel's 10-90% rise time is nearly
    order-invariant above order 2 (it varies <1%). The kind is the bigger lever -- a
    Butterworth is ~8% faster than a Bessel at the same cutoff and order. What matters is
    that the number FOLLOWS the design instead of being frozen at 8-pole Bessel.
    """
    x = _trace_with_events()

    def rise(**kw):
        return detect_events(
            x, FS, filter_cutoff_hz=3_000.0, min_dwell_s=1e-4, **kw)[1]["filter_rise_s"]

    bessel_8 = rise(filter_kind="bessel", filter_order=8)
    butter_8 = rise(filter_kind="butter", filter_order=8)
    gaussian = rise(filter_kind="gaussian")
    bessel_1 = rise(filter_kind="bessel", filter_order=1)

    assert butter_8 != pytest.approx(bessel_8, rel=1e-3)
    assert gaussian != pytest.approx(bessel_8, rel=1e-3)
    assert bessel_1 != pytest.approx(bessel_8, rel=1e-3)


def test_provenance_records_the_design_so_a_result_can_be_reproduced():
    """`filters` says the CALLER stamps filter_kind / filter_order / filter_zero_phase.
    Nothing used to; an exported EventSet could not be reproduced from its own record."""
    x = _trace_with_events()
    _, prov = detect_events(
        x, FS, filter_cutoff_hz=8_000.0, min_dwell_s=1e-4,
        filter_kind="butter", filter_order=3)
    assert prov["filter_cutoff_hz"] == 8_000.0
    assert prov["filter_kind"] == "butter"
    assert prov["filter_order"] == 3
    assert prov["filter_zero_phase"] is True


def test_no_lowpass_records_no_design_at_all():
    x = _trace_with_events()
    _, prov = detect_events(x, FS, min_dwell_s=1e-4)
    assert prov["filter_cutoff_hz"] is None
    assert prov["filter_kind"] is None
    assert prov["filter_order"] is None
    assert prov["filter_zero_phase"] is None


# ================================================================================================
# preprocess_for_detection: hp_order / hp_kind and the detrend shape parameters
# ================================================================================================

def test_high_pass_order_and_kind_reach_the_kernel():
    x = _trace_with_events()
    base = {"hp_enabled": True, "hp_cutoff_hz": 500.0}
    default = preprocess_for_detection(x, FS, base)
    bessel_8 = preprocess_for_detection(
        x, FS, {**base, "hp_kind": "bessel", "hp_order": 8})
    assert not np.allclose(default, bessel_8)
    # The default keys reproduce the frozen 4-pole Butterworth exactly.
    explicit_default = preprocess_for_detection(
        x, FS, {**base, "hp_kind": "butter", "hp_order": 4})
    assert np.array_equal(default, explicit_default)


def test_high_pass_keys_are_registered_so_they_never_reach_the_detector():
    """An unregistered preprocessing key becomes a TypeError inside detect_events."""
    for key in ("hp_order", "hp_kind"):
        assert key in FILTER_PARAM_KEYS
    for key in ("detrend_percentile", "detrend_asls_lam", "detrend_asls_p",
                "detrend_asls_niter", "detrend_asls_decim"):
        assert key in DETREND_PARAM_KEYS
    # The low-pass design is the exception: it IS a detect_events kwarg.
    assert "filter_order" not in FILTER_PARAM_KEYS
    assert "filter_kind" not in FILTER_PARAM_KEYS


def test_detrend_method_kwargs_forwards_only_the_active_methods_parameters():
    params = {
        "detrend_percentile": 90.0, "detrend_asls_lam": 1e5,
        "detrend_asls_p": 0.01, "detrend_asls_niter": 20, "detrend_asls_decim": 0.2,
    }
    # block_median has no shape parameters -> the call is exactly what it always was.
    assert detrend_method_kwargs({**params, "detrend_method": "block_median"}) == {}
    assert detrend_method_kwargs({**params, "detrend_method": "percentile"}) == {
        "percentile": 90.0}
    assert detrend_method_kwargs({**params, "detrend_method": "asls"}) == {
        "asls_lam": 1e5, "asls_p": 0.01, "asls_niter": 20, "asls_decim": 0.2}


def test_detrend_method_kwargs_falls_back_to_the_kernel_defaults():
    """An old params dict (a restored project) must reproduce the frozen values."""
    assert detrend_method_kwargs({"detrend_method": "percentile"}) == {"percentile": 80.0}
    assert detrend_method_kwargs({"detrend_method": "asls"}) == {
        "asls_lam": 6e4, "asls_p": 1e-3, "asls_niter": 10, "asls_decim": 0.1}


def test_percentile_is_no_longer_a_dead_knob():
    x = _trace_with_events()
    base = {"detrend_enabled": True, "detrend_method": "percentile",
            "detrend_window_s": 0.05, "polarity": 1}
    frozen = preprocess_for_detection(x, FS, base)
    moved = preprocess_for_detection(x, FS, {**base, "detrend_percentile": 20.0})
    assert not np.allclose(frozen, moved)
    # The absent key still means 80.0, exactly as before it was exposed.
    assert np.array_equal(frozen, preprocess_for_detection(
        x, FS, {**base, "detrend_percentile": 80.0}))


def test_asls_smoothness_is_no_longer_a_dead_knob():
    x = _trace_with_events()
    base = {"detrend_enabled": True, "detrend_method": "asls",
            "detrend_window_s": 0.05, "polarity": 1}
    frozen = preprocess_for_detection(x, FS, base)
    stiffer = preprocess_for_detection(x, FS, {**base, "detrend_asls_lam": 1e8})
    assert not np.allclose(frozen, stiffer)
    assert np.array_equal(frozen, preprocess_for_detection(
        x, FS, {**base, "detrend_asls_lam": 6e4, "detrend_asls_p": 1e-3,
                "detrend_asls_niter": 10, "detrend_asls_decim": 0.1}))


def test_block_median_ignores_the_other_methods_stale_parameters():
    """Leftovers from another method must not leak into the default estimator."""
    x = _trace_with_events()
    base = {"detrend_enabled": True, "detrend_method": "block_median",
            "detrend_window_s": 0.05, "polarity": 1}
    clean = preprocess_for_detection(x, FS, base)
    littered = preprocess_for_detection(
        x, FS, {**base, "detrend_percentile": 5.0, "detrend_asls_lam": 1e9})
    assert np.array_equal(clean, littered)
