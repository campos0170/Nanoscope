"""Phase 5 seam-adapter tests (PORT_PLAN §4, D3).

Covers:
  * DetectSchmitt registers into ANALYSIS_REGISTRY (register_all is idempotent).
  * Running it on a synthetic CalibratedChannel yields an AnalysisResult with the
    right shape: EventSet payload, one SPAN OverlayHint whose int64 start/stop
    indices match the event table, and the injected calibration_fingerprint.
  * requires_time_base is honoured: no sample rate -> RuntimeError.
  * Session.add_result stores into the ResultStore and emits resultsChanged.
  * The controller-invoked fit wrappers (fit_dwell/fit_depth) return derived
    AnalysisResults carrying a FitResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import (
    DETREND_PARAM_KEYS,
    FILTER_PARAM_KEYS,
    DetectSchmitt,
    _detect_kwargs,
    any_preprocess_enabled,
    fit_depth,
    fit_dwell,
    preprocess_for_detection,
    register_all,
)
from nanoscope.analysis.stats import FitResult
from nanoscope.core.analysis import ANALYSIS_REGISTRY, AnalysisResult, OverlayKind
from nanoscope.core.types import ChannelKind

FS = 100_000.0
# Five square blockades (downward excursions) on a flat, noisy baseline.
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


def make_event_trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return x.astype(np.float32)


# A drifting variant: the same five events on a slowly wandering open pore, for the
# baseline-wander-removal (detrend) tests. n=300_000 -> 3 s at FS, so a 0.3 s detrend
# window forms ~10 blocks — enough to track the drift.
_DRIFT_EVENTS = [(30_000, 30_300), (90_000, 90_250), (150_000, 150_400),
                 (210_000, 210_200), (270_000, 270_350)]


def make_drift_trace(n: int = 300_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(current, event_mask): five blockades on a linear+sinusoidal drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    drift = 45.0 * np.sin(2 * np.pi * 0.5 * t) + 30.0 * (t / t[-1])
    x = 100.0 + drift + rng.normal(0.0, 1.5, n)
    ev = np.zeros(n, dtype=bool)
    for s, e in _DRIFT_EVENTS:
        x[s:e] -= 30.0
        ev[s:e] = True
    return x.astype(np.float32), ev


@pytest.fixture(autouse=True)
def _ensure_registered():
    register_all()  # idempotent — safe even though the qapp fixture also calls it
    yield


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def test_module_registers():
    assert "detect.schmitt" in ANALYSIS_REGISTRY
    m = ANALYSIS_REGISTRY["detect.schmitt"]
    assert m.id == "detect.schmitt"
    assert m.requires_time_base is True
    assert ChannelKind.CURRENT in m.accepts_kinds


def test_register_all_idempotent():
    register_all()
    register_all()
    assert list(ANALYSIS_REGISTRY).count("detect.schmitt") == 1


# ---------------------------------------------------------------------------
# run() shape + overlays + fingerprint
# ---------------------------------------------------------------------------
def test_run_produces_result_shape(context_factory):
    ctx = context_factory(
        make_event_trace(),
        fs=FS,
        params={
            "k_in": 5.0,
            "k_out": 2.0,
            "min_dwell_s": 5e-4,
            "pore_id": "P1",
            "_calibration_fingerprint": (2.5, 1.0),
        },
        units="pA",
        path="pore_p1.dat",
    )
    res = DetectSchmitt().run(ctx)

    assert isinstance(res, AnalysisResult)
    assert res.module_id == "detect.schmitt"
    assert res.module_version == "1.0"
    assert res.channel == 0
    assert res.doc_id.endswith("pore_p1.dat")

    # payload is the EventSet, all five events recovered
    assert isinstance(res.payload, EventSet)
    assert len(res.payload) == len(_EVENTS)
    assert (res.payload.df["pore_id"] == "P1").all()

    # calibration fingerprint injected by the controller is carried verbatim
    assert res.calibration_fingerprint == (2.5, 1.0)
    assert all(isinstance(v, float) for v in res.calibration_fingerprint)

    # scalars
    assert res.scalars["n_events"] == float(len(_EVENTS))
    assert res.scalars["rate_hz"] > 0.0
    assert res.scalars["noise_rms"] > 0.0


def test_run_overlay_indices_match_table(context_factory):
    ctx = context_factory(
        make_event_trace(), fs=FS,
        params={"min_dwell_s": 5e-4, "_calibration_fingerprint": (1.0, 0.0)},
    )
    res = DetectSchmitt().run(ctx)

    assert len(res.overlays) == 1
    ov = res.overlays[0]
    assert ov.kind is OverlayKind.SPAN
    assert ov.starts.dtype == np.int64
    assert ov.stops.dtype == np.int64
    assert ov.starts.size == len(_EVENTS)

    df = res.payload.df
    np.testing.assert_array_equal(ov.starts, df["start_idx"].to_numpy(np.int64))
    np.testing.assert_array_equal(ov.stops, df["end_idx"].to_numpy(np.int64))
    # starts land near the placed events (within a couple of samples of the edge)
    for got, (want, _) in zip(sorted(ov.starts), _EVENTS):
        assert abs(int(got) - want) <= 5


def test_run_default_fingerprint_when_not_injected(context_factory):
    ctx = context_factory(make_event_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    res = DetectSchmitt().run(ctx)
    assert res.calibration_fingerprint == (1.0, 0.0)


def test_requires_time_base_no_rate_raises(context_factory):
    ctx = context_factory(make_event_trace(n=4_000), fs=None, params={})
    with pytest.raises(RuntimeError, match="sample rate"):
        DetectSchmitt().run(ctx)


def test_progress_reported(context_factory):
    seen: list[tuple[float, str]] = []
    ctx = context_factory(make_event_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    ctx.progress_cb = lambda f, m="": seen.append((f, m))
    DetectSchmitt().run(ctx)
    assert seen[0][0] == 0.0
    assert seen[-1][0] == 1.0


# ---------------------------------------------------------------------------
# Session.add_result
# ---------------------------------------------------------------------------
def test_add_result_stores_and_emits(qapp, context_factory):
    from nanoscope.core.session import Session

    ctx = context_factory(make_event_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    res = DetectSchmitt().run(ctx)

    sess = Session()
    seen: list[tuple[str, str]] = []
    sess.resultsChanged.connect(lambda d, m: seen.append((d, m)))
    sess.add_result(res)

    assert seen == [(res.doc_id, res.module_id)]
    stored = sess.results.get(res.doc_id, res.channel, res.module_id)
    assert stored is res


# ---------------------------------------------------------------------------
# fit wrappers (D3 — controller-invoked, not modules)
# ---------------------------------------------------------------------------
def test_fit_dwell_wrapper_returns_result():
    rng = np.random.default_rng(0)
    t_min = 2e-5
    dwell = t_min + rng.exponential(5e-4, 200)
    df = pd.DataFrame({"dwell_s": dwell, "accepted": True})

    res = fit_dwell(df, doc_id="doc-1", channel=0, t_min=t_min,
                    calibration_fingerprint=(2.0, 0.0))

    assert isinstance(res, AnalysisResult)
    assert res.module_id == "fit.dwell"
    assert res.doc_id == "doc-1"
    assert res.calibration_fingerprint == (2.0, 0.0)
    assert isinstance(res.payload, FitResult)
    assert "tau_1" in res.payload.params
    assert res.scalars["n"] == 200.0
    # recovered tau near the true 5e-4 s scale
    assert 2e-4 < res.payload.params["tau_1"] < 1e-3


def test_fit_dwell_infers_t_min_from_eventset():
    # Build an EventSet whose provenance carries min_dwell_s -> t_min inferred.
    rng = np.random.default_rng(1)
    t_min = 3e-5
    dwell = t_min + rng.exponential(4e-4, 120)
    from nanoscope.analysis.events import empty_df

    df = empty_df()
    df["dwell_s"] = dwell
    df["accepted"] = True
    df["frac_depth"] = 0.1 + rng.normal(0.0, 0.01, 120)
    es = EventSet(df, {"min_dwell_s": t_min, "fs": FS, "analyzed_duration_s": 1.0})

    res = fit_dwell(es, doc_id="d", channel=0)
    assert res.params["t_min"] == t_min
    assert isinstance(res.payload, FitResult)


def test_fit_depth_wrapper_returns_result():
    rng = np.random.default_rng(2)
    frac = np.abs(rng.normal(0.2, 0.02, 200))
    df = pd.DataFrame({"frac_depth": frac, "accepted": True})

    res = fit_depth(df, doc_id="doc-2", channel=1, calibration_fingerprint=(1.0, 0.0))

    assert res.module_id == "fit.depth"
    assert res.channel == 1
    assert isinstance(res.payload, FitResult)
    assert "mu_1" in res.payload.params


# ---------------------------------------------------------------------------
# baseline-wander removal (detrend) preprocessing seam
# ---------------------------------------------------------------------------
def test_preprocess_passthrough_when_detrend_disabled():
    x = make_event_trace()
    out = preprocess_for_detection(x, FS, {"detrend_enabled": False})
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)  # untouched


def test_preprocess_passthrough_when_rate_unknown():
    """No time base -> the detrend estimators are undefined; return the signal as-is
    (detection is gated off in this case anyway)."""
    x = make_event_trace()
    out = preprocess_for_detection(x, None, {"detrend_enabled": True})
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_preprocess_refuses_nonfinite_samples_before_a_filter_can_hide_them(bad):
    """A median can replace one invalid sample with finite neighbours, so integrity
    is checked at the preprocessing entrance rather than after the chain."""
    x = np.linspace(-1.0, 1.0, 21)
    x[10] = bad
    with pytest.raises(ValueError, match="NaN/Inf before preprocessing"):
        preprocess_for_detection(
            x, FS, {"median_enabled": True, "median_window_s": 5 / FS})


def test_preprocess_refuses_values_that_overflow_its_float32_domain():
    with pytest.raises(ValueError, match="float32 processing range"):
        preprocess_for_detection(np.array([0.0, 1e300]), FS, {})


def test_preprocess_refuses_non_vector_input_at_its_boundary():
    with pytest.raises(ValueError, match="must be 1-D before preprocessing"):
        preprocess_for_detection(np.zeros((2, 8)), FS, {})


def test_preprocess_flattens_drift_when_enabled():
    """The signal the detector actually sees is flatter in the open pore."""
    x, ev = make_drift_trace()
    out = preprocess_for_detection(
        x, FS,
        {"detrend_enabled": True, "detrend_method": "block_median",
         "detrend_window_s": 0.3, "polarity": None},
    )
    assert out.dtype == np.float32
    assert out.shape == x.shape
    assert out[~ev].std() < x[~ev].std() / 4.0


def test_detrend_keys_are_not_forwarded_to_detector():
    """_detect_kwargs must strip the detrend keys — detect_events has no such params
    and would raise a TypeError if handed them."""
    params = {
        "k_in": 5.0, "min_dwell_s": 5e-4,
        "detrend_enabled": True, "detrend_method": "percentile", "detrend_window_s": 0.4,
    }
    fwd = _detect_kwargs(params)
    assert not (set(DETREND_PARAM_KEYS) & set(fwd))
    assert fwd["k_in"] == 5.0 and fwd["min_dwell_s"] == 5e-4


def test_run_with_detrend_records_provenance_and_finds_events(context_factory):
    x, _ev = make_drift_trace()
    ctx = context_factory(
        x, fs=FS,
        params={"min_dwell_s": 5e-4, "detrend_enabled": True,
                "detrend_method": "block_median", "detrend_window_s": 0.3},
    )
    res = DetectSchmitt().run(ctx)

    prov = res.payload.provenance
    assert prov["detrend_enabled"] is True
    assert prov["detrend_method"] == "block_median"
    assert prov["detrend_window_s"] == pytest.approx(0.3)
    # Detection still recovers every event on the flattened trace.
    assert len(res.payload) == len(_DRIFT_EVENTS)


def test_run_without_detrend_marks_provenance_false(context_factory):
    ctx = context_factory(make_event_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    res = DetectSchmitt().run(ctx)
    prov = res.payload.provenance
    assert prov["detrend_enabled"] is False
    # No method/window keys are stamped when detrend is off (nothing to describe).
    assert "detrend_method" not in prov
    assert "detrend_window_s" not in prov


# ---------------------------------------------------------------------------
# the rest of the pre-detection filter chain: notch -> high-pass -> savgol
# ---------------------------------------------------------------------------
def test_any_preprocess_enabled_flags_the_stages_not_the_lowpass():
    assert not any_preprocess_enabled({})
    # The low-pass is applied later by detect_events, so it is NOT a preprocess stage.
    assert not any_preprocess_enabled({"filter_cutoff_hz": 10_000.0})
    for flag in ("notch_enabled", "hp_enabled", "detrend_enabled", "savgol_enabled"):
        assert any_preprocess_enabled({flag: True})
        assert not any_preprocess_enabled({flag: False})


def test_filter_keys_are_not_forwarded_to_detector():
    """_detect_kwargs must strip every FILTER_PARAM_KEYS entry — detect_events has no
    such parameters and would raise a TypeError if handed them."""
    params = {
        "k_in": 5.0, "min_dwell_s": 5e-4,
        "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 3, "notch_q": 30.0,
        "hp_enabled": True, "hp_cutoff_hz": 100.0,
        "savgol_enabled": True, "savgol_window_s": 1e-3, "savgol_polyorder": 3,
    }
    fwd = _detect_kwargs(params)
    assert not (set(FILTER_PARAM_KEYS) & set(fwd))
    assert fwd["k_in"] == 5.0 and fwd["min_dwell_s"] == 5e-4


def test_preprocess_passthrough_when_no_stage_enabled():
    x = make_event_trace()
    out = preprocess_for_detection(x, FS, {"k_in": 5.0})  # detection params, no filter flags
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)


def test_preprocess_notch_removes_mains_hum():
    from nanoscope.analysis.filters import psd

    fs = 50_000.0
    n = 200_000
    t = np.arange(n) / fs
    x = (np.random.default_rng(0).normal(0, 1, n)
         + 40 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    out = preprocess_for_detection(x, fs, {"notch_enabled": True, "notch_harmonics": 1})
    f0, p0 = psd(x, fs, nperseg=65536)
    f1, p1 = psd(out, fs, nperseg=65536)

    def power_near(f, p, target):
        return p[np.argmin(np.abs(f - target))]

    assert power_near(f1, p1, 60) < 1e-2 * power_near(f0, p0, 60)


def test_preprocess_highpass_removes_dc_offset():
    """The high-pass stage strips a DC pedestal; passthrough would keep it."""
    x = make_event_trace() + 500.0  # a large DC pedestal on top of the events
    out = preprocess_for_detection(x, FS, {"hp_enabled": True, "hp_cutoff_hz": 50.0})
    assert abs(float(np.mean(x))) > 400.0
    assert abs(float(np.mean(out))) < 20.0


def test_preprocess_savgol_smooths_the_open_pore_noise():
    x = make_event_trace()  # flat 100 pA + 1.5 pA noise + events
    out = preprocess_for_detection(x, FS, {"savgol_enabled": True, "savgol_window_s": 3e-4})
    assert out.dtype == np.float32
    # The open-pore noise floor (pre-first-event) shrinks; peak shapes are preserved.
    assert out[:9_000].std() < 0.7 * x[:9_000].std()


def test_preprocess_applies_the_full_chain_and_returns_float32():
    """notch + detrend + savgol together flatten and smooth the open pore. (High-pass is
    left off here: it distorts these ms-scale events by design — see its own tests.)"""
    x, ev = make_drift_trace()
    out = preprocess_for_detection(
        x, FS,
        {"notch_enabled": True, "notch_harmonics": 1,
         "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.3,
         "savgol_enabled": True, "savgol_window_s": 2e-4, "savgol_polyorder": 3},
    )
    assert out.dtype == np.float32
    assert out.shape == x.shape
    assert out[~ev].std() < x[~ev].std() / 4.0


def test_run_with_notch_records_provenance_and_finds_all_events(context_factory):
    """Notch is nearly free for these fast events, so every event survives, and the
    result records the transform."""
    ctx = context_factory(
        make_event_trace(), fs=FS,
        params={"min_dwell_s": 5e-4, "notch_enabled": True,
                "notch_freq_hz": 60.0, "notch_harmonics": 2, "notch_q": 25.0},
    )
    res = DetectSchmitt().run(ctx)
    prov = res.payload.provenance
    assert prov["notch_enabled"] is True
    assert prov["notch_freq_hz"] == pytest.approx(60.0)
    assert prov["notch_harmonics"] == 2
    assert prov["notch_q"] == pytest.approx(25.0)
    assert len(res.payload) == len(_EVENTS)


def test_run_records_highpass_and_savgol_provenance(context_factory):
    """Provenance is stamped whether or not the filters change the event count (a
    high-pass legitimately alters detection by making square events biphasic)."""
    ctx = context_factory(
        make_event_trace(), fs=FS,
        params={"min_dwell_s": 5e-4,
                "hp_enabled": True, "hp_cutoff_hz": 40.0,
                "savgol_enabled": True, "savgol_window_s": 2e-4, "savgol_polyorder": 3},
    )
    res = DetectSchmitt().run(ctx)
    prov = res.payload.provenance
    assert prov["hp_enabled"] is True
    assert prov["hp_cutoff_hz"] == pytest.approx(40.0)
    assert prov["savgol_enabled"] is True
    assert prov["savgol_window_s"] == pytest.approx(2e-4)
    assert prov["savgol_polyorder"] == 3


def test_run_without_filters_marks_all_provenance_false(context_factory):
    ctx = context_factory(make_event_trace(), fs=FS, params={"min_dwell_s": 5e-4})
    prov = DetectSchmitt().run(ctx).payload.provenance
    for flag in ("notch_enabled", "median_enabled", "hp_enabled",
                 "rmean_enabled", "savgol_enabled"):
        assert prov[flag] is False
    # No parameter keys are stamped for a disabled stage (nothing to describe).
    for k in ("notch_freq_hz", "notch_harmonics", "notch_q",
              "median_window_s", "hp_cutoff_hz", "rmean_window_s",
              "savgol_window_s", "savgol_polyorder"):
        assert k not in prov


# ---------------------------------------------------------------------------
# running mean + median: the two smoothing/despiking stages
# ---------------------------------------------------------------------------
def test_any_preprocess_enabled_flags_median_and_rmean():
    for flag in ("median_enabled", "rmean_enabled"):
        assert any_preprocess_enabled({flag: True})
        assert not any_preprocess_enabled({flag: False})


def test_median_and_rmean_keys_are_not_forwarded_to_detector():
    """Both new stages' keys are in FILTER_PARAM_KEYS, so _detect_kwargs strips them —
    detect_events has no such parameters and would raise a TypeError if handed them."""
    params = {
        "k_in": 5.0, "min_dwell_s": 5e-4,
        "median_enabled": True, "median_window_s": 1e-4,
        "rmean_enabled": True, "rmean_window_s": 5e-4,
    }
    fwd = _detect_kwargs(params)
    assert "median_enabled" not in fwd and "median_window_s" not in fwd
    assert "rmean_enabled" not in fwd and "rmean_window_s" not in fwd
    assert fwd["k_in"] == 5.0 and fwd["min_dwell_s"] == 5e-4


def test_preprocess_median_removes_spikes_preserving_events():
    """The median stage deletes single-sample impulses while leaving the real events
    detectable (their depth and edges survive)."""
    x = make_event_trace().copy()
    x[5_000] += 500.0  # a lone impulse in the open pore
    x[20_000] -= 500.0
    out = preprocess_for_detection(x, FS, {"median_enabled": True, "median_window_s": 5e-5})
    assert out.dtype == np.float32
    # The spikes are gone (open-pore floor is ~100 pA, not ~±400 pA).
    assert abs(float(out[5_000]) - 100.0) < 10.0
    assert abs(float(out[20_000]) - 100.0) < 10.0
    # A real event's depth is preserved (~70 pA inside a 30 pA blockade).
    assert float(out[10_150]) == pytest.approx(70.0, abs=3.0)


def test_preprocess_rmean_smooths_the_open_pore_noise():
    x = make_event_trace()  # flat 100 pA + 1.5 pA noise + events
    out = preprocess_for_detection(x, FS, {"rmean_enabled": True, "rmean_window_s": 2e-4})
    assert out.dtype == np.float32
    # The open-pore noise floor (pre-first-event) shrinks.
    assert out[:9_000].std() < 0.5 * x[:9_000].std()


def test_run_records_median_and_rmean_provenance(context_factory):
    ctx = context_factory(
        make_event_trace(), fs=FS,
        params={"min_dwell_s": 5e-4,
                "median_enabled": True, "median_window_s": 5e-5,
                "rmean_enabled": True, "rmean_window_s": 2e-4},
    )
    prov = DetectSchmitt().run(ctx).payload.provenance
    assert prov["median_enabled"] is True
    assert prov["median_window_s"] == pytest.approx(5e-5)
    assert prov["rmean_enabled"] is True
    assert prov["rmean_window_s"] == pytest.approx(2e-4)


# ===========================================================================
# N2 — the trace a re-characterization must measure on.
#
# ``DetectSchmitt`` conditions the signal (``preprocess_for_detection``) and then
# ``detect_events`` applies its own low-pass, so every ``depth_pA`` / ``baseline_pA`` /
# ``rise_s`` in the table was read off ``lowpass(preprocess_for_detection(raw))``.
# Anything that re-measures an event afterwards -- a merge, an Event Inspector mask
# resize, a hand-added missed event -- was reading ``ch.data``, the RAW channel, and
# writing the answer into the same columns.
#
# Measured on the user's -600 mV recording over 349 merged spans: ``frac_depth`` came
# out 3.3% off at the median at the app's default 10 kHz Bessel (p90 8.0%), 10.7% at
# 2 kHz (p90 47.5%), and one row changed sign.
#
# ``detection_signal`` rebuilds that trace from the detection's own provenance, which
# ``DetectSchmitt`` stamps for exactly this purpose.
# ===========================================================================

def test_detection_signal_reproduces_the_detectors_own_trace_bit_for_bit(context_factory):
    """The whole claim in one assertion, through the REAL module: what the detector
    measured on, rebuilt from nothing but the result it produced."""
    from nanoscope.analysis.filters import lowpass
    from nanoscope.analysis.modules import detection_signal

    x, _ev = make_drift_trace()
    params = {
        "detrend_enabled": True, "detrend_method": "block_median",
        "detrend_window_s": 0.3, "polarity": -1,
        "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 2,
        "notch_q": 30.0,
        "savgol_enabled": True, "savgol_window_s": 1e-3, "savgol_polyorder": 3,
        "filter_cutoff_hz": 10_000.0, "filter_kind": "bessel", "filter_order": 8,
        "k_in": 5.0, "k_out": 2.0,
    }
    res = DetectSchmitt().run(context_factory(x, fs=FS, params=params))
    prov = res.payload.provenance

    want = lowpass(preprocess_for_detection(x, FS, params), FS,
                   10_000.0, order=8, kind="bessel")
    got = detection_signal(x, FS, prov)
    np.testing.assert_array_equal(got, want)
    # ...and it is emphatically NOT the raw channel.
    assert float(np.abs(np.asarray(got, np.float64) - x).max()) > 1.0


def test_detection_signal_carries_the_detrend_methods_own_shape_parameters(context_factory):
    """Provenance stamps the detrend shape knobs under their KERNEL names
    (``percentile``) while ``preprocess_for_detection`` reads them under their PARAM
    names (``detrend_percentile``). Feeding provenance back in unmapped would silently
    fall back to the kernel default for exactly the knob the user moved."""
    from nanoscope.analysis.modules import (
        detection_signal, preprocess_params_from_provenance,
    )

    x, _ev = make_drift_trace()
    params = {"detrend_enabled": True, "detrend_method": "percentile",
              "detrend_window_s": 0.3, "detrend_percentile": 95.0, "polarity": -1}
    res = DetectSchmitt().run(context_factory(x, fs=FS, params=params))
    prov = res.payload.provenance

    assert prov["percentile"] == 95.0              # stamped under the kernel name
    assert "detrend_percentile" not in prov
    back = preprocess_params_from_provenance(prov)
    assert back["detrend_percentile"] == 95.0      # ...translated back

    np.testing.assert_array_equal(
        detection_signal(x, FS, prov), preprocess_for_detection(x, FS, params))
    # The default (80.0) would give a different trace, which is what the unmapped
    # version silently used.
    other = dict(params, detrend_percentile=80.0)
    assert not np.array_equal(
        detection_signal(x, FS, prov), preprocess_for_detection(x, FS, other))


def test_detection_signal_leaves_a_provenance_that_describes_nothing_alone():
    """An old project / a hand-built EventSet / an ML run with its own chain records no
    preprocessing; guessing at one would be worse than not conditioning at all."""
    from nanoscope.analysis.modules import detection_signal

    x = make_event_trace()
    np.testing.assert_array_equal(detection_signal(x, FS, {}), x)
    np.testing.assert_array_equal(detection_signal(x, None, {"filter_cutoff_hz": 1e4}), x)
    # A cutoff that cannot be applied is skipped rather than raised through.
    np.testing.assert_array_equal(
        detection_signal(x, FS, {"filter_cutoff_hz": FS}), x)


def test_detection_signal_key_changes_with_every_field_it_reads():
    """The cache key the GUI panels use: if it misses a field, a re-detection with new
    filtering would keep serving the old conditioned trace."""
    from nanoscope.analysis.modules import detection_signal_key

    base = {"filter_cutoff_hz": 1e4, "filter_kind": "bessel", "filter_order": 8,
            "polarity": -1, "detrend_enabled": True, "detrend_method": "percentile",
            "detrend_window_s": 0.3, "percentile": 95.0, "notch_enabled": False}
    for field, value in (("filter_cutoff_hz", 2e3), ("filter_kind", "butter"),
                         ("filter_order", 4), ("polarity", 1),
                         ("detrend_enabled", False), ("detrend_window_s", 0.1),
                         ("percentile", 80.0), ("notch_enabled", True)):
        assert detection_signal_key(dict(base, **{field: value})) != \
            detection_signal_key(base), field


def test_detection_signal_uses_the_polarity_the_user_asked_for_not_the_resolved_one(
    context_factory,
):
    """``preprocess_for_detection`` reads the polarity the USER set (often None, which
    makes the detrend derive it from the unfiltered trace); ``provenance['polarity']`` is
    what ``detect_events`` resolved afterwards from the LOW-PASSED trace. Reconstructing
    from the resolved one would rebuild a detrend that never ran."""
    from nanoscope.analysis.modules import (
        detection_signal, preprocess_params_from_provenance,
    )

    x, _ev = make_drift_trace()
    params = {"detrend_enabled": True, "detrend_method": "block_median",
              "detrend_window_s": 0.3, "polarity": None,
              "filter_cutoff_hz": 10_000.0, "filter_kind": "bessel", "filter_order": 8}
    res = DetectSchmitt().run(context_factory(x, fs=FS, params=params))
    prov = res.payload.provenance

    assert prov["polarity_requested"] is None
    assert prov["polarity"] in (-1, 1)                  # resolved, and NOT what to reuse
    assert preprocess_params_from_provenance(prov)["polarity"] is None

    from nanoscope.analysis.filters import lowpass
    want = lowpass(preprocess_for_detection(x, FS, params), FS,
                   10_000.0, order=8, kind="bessel")
    np.testing.assert_array_equal(detection_signal(x, FS, prov), want)


def test_the_normalize_stage_is_actually_applied_and_not_forwarded_to_the_detector():
    """The Normalize page shipped as a checkbox wired to nothing: `filters.normalize`
    existed, `DetectionPanel.params()` emitted the two keys, and the chain never called
    it — so the stage was a no-op that additionally raised
    ``detect_events() got an unexpected keyword argument 'normalize_enabled'``.

    Both halves are pinned here: the keys are CONSUMED (never forwarded), and the trace
    the detector sees is genuinely rescaled.
    """
    from nanoscope.analysis.modules import (
        _CONSUMED_PARAM_KEYS,
        any_preprocess_enabled,
        preprocess_for_detection,
    )

    assert {"normalize_enabled", "normalize_mode"} <= set(FILTER_PARAM_KEYS)
    assert {"normalize_enabled", "normalize_mode"} <= _CONSUMED_PARAM_KEYS
    assert not _detect_kwargs({"normalize_enabled": True, "normalize_mode": "zscore"})

    rng = np.random.default_rng(0)
    x = (rng.normal(0.0, 10.0, 50_000) + 300.0).astype(np.float32)
    params = {"normalize_enabled": True, "normalize_mode": "robust"}

    assert any_preprocess_enabled(params)
    out = preprocess_for_detection(x, FS, params)
    assert abs(float(out.std()) - 1.0) < 0.05        # sigmas, not picoamps
    assert abs(float(np.median(out))) < 0.05

    # Disabled is the identity, and an unknown rate still applies it (no time base needed).
    np.testing.assert_array_equal(preprocess_for_detection(x, FS, {}), x)
    np.testing.assert_allclose(preprocess_for_detection(x, None, params), out, rtol=1e-6)


def test_normalize_runs_last_so_the_smoothers_are_unchanged_by_it():
    """Stage order is the whole reason the six conditioning stages still measure what
    they measured before Normalize existed: a single affine map applied AFTER them
    commutes out of every ratio the disclosures report."""
    from nanoscope.analysis.filters import normalize
    from nanoscope.analysis.modules import preprocess_for_detection

    rng = np.random.default_rng(1)
    x = (rng.normal(0.0, 5.0, 20_000) + 250.0).astype(np.float32)
    chain = {"rmean_enabled": True, "rmean_window_s": 1e-4}

    smoothed = preprocess_for_detection(x, FS, chain)
    both = preprocess_for_detection(x, FS, dict(chain, normalize_enabled=True,
                                                normalize_mode="zscore"))
    np.testing.assert_allclose(both, normalize(smoothed, mode="zscore"), rtol=1e-5)


# ---------------------------------------------------------------------------
# ROUND 5: the kernels that were implemented but unreachable, and the second detector
# ---------------------------------------------------------------------------
def test_the_explicit_notch_line_list_reaches_the_kernel():
    """``notch_freqs`` REPLACES the (freq, harmonics) series -- the series values are
    not merged, not used as a seed, and not read at all.

    Pinned by comparing the chain's output against a direct kernel call made with the
    line list and DELIBERATELY ABSURD series values. If preprocess_for_detection were
    merging the two spellings (or falling back to the series), the two arrays would
    differ; they are identical to the bit.
    """
    from nanoscope.analysis.filters import notch
    from nanoscope.analysis.modules import preprocess_for_detection

    rng = np.random.default_rng(3)
    t = np.arange(50_000) / FS
    x = (rng.normal(0.0, 2.0, t.size) + 5.0 * np.sin(2 * np.pi * 300.0 * t)).astype(
        np.float32)
    lines = [60.0, 120.0, 180.0, 300.0, 420.0, 540.0]

    got = preprocess_for_detection(x, FS, {
        "notch_enabled": True,
        "notch_freqs": lines,
        "notch_q": 30.0,
        # If these were read at all the answer would be a 17-line comb at 999 Hz.
        "notch_freq_hz": 999.0,
        "notch_harmonics": 17,
    })
    want = notch(x, FS, freq_hz=999.0, q=30.0, harmonics=17, freqs=lines)
    np.testing.assert_array_equal(got, want)


def test_a_notch_line_list_is_a_key_the_module_consumes_not_a_detector_kwarg():
    from nanoscope.analysis.modules import _CONSUMED_PARAM_KEYS

    assert "notch_freqs" in FILTER_PARAM_KEYS
    assert "notch_freqs" in _CONSUMED_PARAM_KEYS
    assert not _detect_kwargs({"notch_freqs": [60.0], "normalize_clip_sigma": 30.0})


def test_notch_settling_is_sized_from_the_lines_that_actually_ran():
    """A nine-line comb rings longer than the three-line default, and the report has to
    know which one ran. Sizing the pad from a filter that did not run is the whole
    failure this guards against."""
    from nanoscope.analysis.modules import notch_settling_from_provenance

    base = {"notch_enabled": True, "fs": FS, "n_samples": 200_000, "notch_q": 30.0}
    series = notch_settling_from_provenance(dict(base, notch_freq_hz=60.0,
                                                 notch_harmonics=3))
    explicit = notch_settling_from_provenance(dict(
        base, notch_freq_hz=60.0, notch_harmonics=3,
        notch_freqs=[60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0]))
    assert series is not None and explicit is not None
    # Same lowest line, so the ring-down is the same -- what must NOT happen is the
    # explicit list being ignored and a DIFFERENT lowest line reported.
    only_high = notch_settling_from_provenance(dict(
        base, notch_freq_hz=60.0, notch_harmonics=3, notch_freqs=[600.0, 1200.0]))
    assert only_high is not None
    assert only_high[0] < series[0]     # a 600 Hz notch settles far faster than 60 Hz


@pytest.mark.parametrize(
    "method, params, expected",
    [
        ("savgol", {"detrend_sg_poly": 4, "detrend_sg_decim": 25},
         {"sg_poly": 4, "sg_decim": 25}),
        ("masked_median",
         {"detrend_rough_s": 0.5, "detrend_base_s": 0.02, "detrend_mask_sigma": 3.5,
          "detrend_dilate_ms": 0.2, "detrend_decim": 10},
         {"rough_s": 0.5, "base_s": 0.02, "mask_sigma": 3.5, "dilate_ms": 0.2,
          "decim": 10}),
        ("robust_median", {"detrend_decim": 50}, {"decim": 50}),
    ],
)
def test_the_three_newly_reachable_estimators_forward_their_own_shape(
        method, params, expected):
    """Each method's knobs reach `estimate_baseline` under its KERNEL names, and only
    that method's -- a stale key from another method cannot leak in."""
    from nanoscope.analysis.modules import detrend_method_kwargs

    got = detrend_method_kwargs(dict(params, detrend_method=method,
                                     detrend_percentile=42.0, detrend_asls_lam=1.0))
    assert got == expected


@pytest.mark.parametrize("method", ["masked_median", "robust_median"])
def test_decim_zero_is_the_kernels_own_auto_stride(method):
    """0 is the panel-side sentinel for "derive it from the window", which the kernel
    spells ``None``. The tuple mechanism cannot express a None default (it coerces with
    ``int(value)``), so the translation happens once, here -- and it is the DEFAULT, so
    both estimators stay byte-identical to the kernel until a stride is pinned."""
    from nanoscope.analysis.modules import detrend_method_kwargs

    assert detrend_method_kwargs({"detrend_method": method})["decim"] is None
    assert detrend_method_kwargs(
        {"detrend_method": method, "detrend_decim": 0})["decim"] is None
    assert detrend_method_kwargs(
        {"detrend_method": method, "detrend_decim": 10})["decim"] == 10


def test_the_savgol_defaults_are_imported_not_re_spelled():
    """The int-ness of these two is load-bearing: `detrend_method_kwargs` coerces with
    ``int(value) if isinstance(default, int) else float(value)``, and ``sg_decim`` must
    reach the kernel as an int. Re-spelling 3/50 as literals here would also let the
    panel and the kernel drift apart."""
    from nanoscope.analysis.baseline import SAVGOL_DECIM, SAVGOL_POLY
    from nanoscope.analysis.modules import DETREND_METHOD_PARAMS

    spec = dict((p, (k, d)) for p, k, d in DETREND_METHOD_PARAMS["savgol"])
    assert spec["detrend_sg_poly"] == ("sg_poly", SAVGOL_POLY)
    assert spec["detrend_sg_decim"] == ("sg_decim", SAVGOL_DECIM)
    assert isinstance(SAVGOL_POLY, int) and isinstance(SAVGOL_DECIM, int)


def test_model_input_normalize_dispatches_to_the_other_kernel():
    """`filters.normalize` REFUSES 'model_input' -- it is a different function with a
    different contract (bounded, non-affine, falls back on a flat trace where normalize
    raises). `_apply_normalize` is the one place they are told apart, and this pins that
    the stage really reaches `robust_normalize` rather than raising or falling back."""
    from nanoscope.analysis.filters import (
        MODEL_INPUT_MODE, normalize, robust_normalize)
    from nanoscope.analysis.modules import preprocess_for_detection

    rng = np.random.default_rng(4)
    x = (rng.normal(0.0, 8.0, 40_000) - 1000.0).astype(np.float32)
    x[5_000:5_200] -= 400.0                       # one very deep event, past 30 sigma

    got = preprocess_for_detection(x, FS, {
        "normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE})
    np.testing.assert_array_equal(got, robust_normalize(x, clip_sigma=30.0))
    assert float(got.min()) >= -1.0 and float(got.max()) <= 1.0

    with pytest.raises(ValueError):
        normalize(x, mode=MODEL_INPUT_MODE)


def test_the_clip_sigma_is_read_only_by_that_mode():
    from nanoscope.analysis.filters import MODEL_INPUT_MODE, robust_normalize
    from nanoscope.analysis.modules import preprocess_for_detection

    rng = np.random.default_rng(5)
    x = (rng.normal(0.0, 8.0, 20_000)).astype(np.float32)

    got = preprocess_for_detection(x, FS, {
        "normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE,
        "normalize_clip_sigma": 10.0})
    np.testing.assert_array_equal(got, robust_normalize(x, clip_sigma=10.0))

    # The other three modes ignore it entirely -- a clip bound there would be a
    # parameter that did nothing.
    a = preprocess_for_detection(x, FS, {"normalize_enabled": True,
                                         "normalize_mode": "robust"})
    b = preprocess_for_detection(x, FS, {"normalize_enabled": True,
                                         "normalize_mode": "robust",
                                         "normalize_clip_sigma": 10.0})
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# the CUSUM detector as a module
# ---------------------------------------------------------------------------
def test_the_cusum_detector_is_registered_beside_the_schmitt_one():
    from nanoscope.analysis.modules import DetectCusum

    assert "detect.cusum" in ANALYSIS_REGISTRY
    m = ANALYSIS_REGISTRY["detect.cusum"]
    assert m.id == DetectCusum.id == "detect.cusum"
    assert m.label == "Detect events (CUSUM)"
    assert m.requires_time_base is True
    assert ChannelKind.CURRENT in m.accepts_kinds
    register_all()
    assert list(ANALYSIS_REGISTRY).count("detect.cusum") == 1


def test_neither_detectors_private_thresholds_reach_the_other():
    """One form carries both detectors' knobs so switching does not discard the other's
    settings; this strip is what makes that safe. Without it a CUSUM run receives
    ``k_in`` and dies on ``unexpected keyword argument`` -- the same failure the
    ``baseline_`` namespace rule was written for, one detector later."""
    params = {"k_in": 5.0, "k_out": 2.0, "cusum_k": 0.75, "cusum_h": 28.0,
              "polarity": None, "detector": "detect.cusum"}

    schmitt = _detect_kwargs(params, "detect.schmitt")
    cusum = _detect_kwargs(params, "detect.cusum")
    assert set(schmitt) == {"k_in", "k_out", "polarity"}
    assert set(cusum) == {"cusum_k", "cusum_h", "polarity"}


def test_cusum_module_run_returns_the_same_shape_as_the_schmitt_one(context_factory):
    """Same payload type, same SPAN overlay, same scalars, same fingerprint -- so every
    surface downstream (table, inspector, gallery, plots, export) treats a CUSUM row
    exactly like a Schmitt one."""
    from nanoscope.analysis.modules import DetectCusum

    ctx = context_factory(
        make_event_trace(),
        fs=FS,
        params={"cusum_k": 0.75, "cusum_h": 28.0, "min_dwell_s": 5e-4,
                "pore_id": "P1", "_calibration_fingerprint": (2.5, 1.0)},
        units="pA",
        path="pore_p1.dat",
    )
    res = DetectCusum().run(ctx)

    assert res.module_id == "detect.cusum"
    assert isinstance(res.payload, EventSet)
    assert len(res.payload) == len(_EVENTS)
    assert res.calibration_fingerprint == (2.5, 1.0)
    ov = res.overlays[0]
    assert ov.kind is OverlayKind.SPAN
    df = res.payload.df
    np.testing.assert_array_equal(ov.starts, df["start_idx"].to_numpy(np.int64))
    prov = res.payload.provenance
    assert prov["detector"] == "cusum_sequential"    # NOT "cusum" -- that is the fitter
    assert set(res.scalars) == {"n_events", "rate_hz", "noise_rms",
                                "median_baseline", "filter_rise_s"}


def test_cusum_module_stamps_the_same_preprocessing_block_as_schmitt(context_factory):
    """ONE writer for the preprocessing stamp. A stage missing from one detector's
    provenance is an event whose depth `detection_signal` cannot reproduce, which is
    precisely what makes a Schmitt row and a CUSUM row incomparable."""
    from nanoscope.analysis.modules import DetectCusum

    chain = {
        "notch_enabled": True, "notch_freqs": [60.0, 180.0], "notch_q": 30.0,
        "detrend_enabled": True, "detrend_method": "savgol",
        "detrend_window_s": 0.2, "detrend_sg_poly": 3, "detrend_sg_decim": 50,
        "min_dwell_s": 5e-4,
    }
    x = make_event_trace()
    a = DetectSchmitt().run(context_factory(x, fs=FS, params=dict(chain, k_in=5.0,
                                                                 k_out=2.0)))
    b = DetectCusum().run(context_factory(x, fs=FS, params=dict(chain, cusum_k=0.75,
                                                                cusum_h=28.0)))
    pa, pb = a.payload.provenance, b.payload.provenance
    for key in ("notch_enabled", "notch_freqs", "notch_q", "detrend_enabled",
                "detrend_method", "detrend_window_s", "sg_poly", "sg_decim",
                "detrend_effective"):
        assert pa[key] == pb[key], key
    # ...and the same shared baseline, so the depths are directly comparable.
    assert pa["baseline_fingerprint"] == pb["baseline_fingerprint"]


def test_detection_stamps_the_detectors_own_bounds_for_the_inspectors_revert(
        context_factory):
    """Stamped at DETECTION and only there -- it is the one moment the bounds are the
    detector's. Both detectors do it, so Revert works on either kind of row."""
    from nanoscope.analysis.events import has_original_bounds, original_bounds
    from nanoscope.analysis.modules import DetectCusum

    for module, extra in ((DetectSchmitt(), {"k_in": 5.0, "k_out": 2.0}),
                          (DetectCusum(), {"cusum_k": 0.75, "cusum_h": 28.0})):
        res = module.run(context_factory(
            make_event_trace(), fs=FS, params=dict(extra, min_dwell_s=5e-4)))
        df = res.payload.df
        assert has_original_bounds(df)
        row = df.iloc[0]
        assert original_bounds(row) == (int(row["start_idx"]), int(row["end_idx"]))


def test_the_normalize_stage_is_stamped_so_the_trace_can_be_rebuilt(context_factory):
    """`detection_signal` rebuilds the exact array a depth was read off from
    provenance alone. The normalize stage was never stamped, so a run with it enabled
    rebuilt an UNNORMALIZED trace -- silently a different measurement."""
    from nanoscope.analysis.filters import MODEL_INPUT_MODE
    from nanoscope.analysis.modules import preprocess_params_from_provenance

    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE,
                "normalize_clip_sigma": 12.0}))
    prov = res.payload.provenance
    assert prov["normalize_enabled"] is True
    assert prov["normalize_mode"] == MODEL_INPUT_MODE
    assert prov["normalize_clip_sigma"] == 12.0
    back = preprocess_params_from_provenance(prov)
    assert back["normalize_enabled"] and back["normalize_clip_sigma"] == 12.0


def test_provenance_stamps_the_lines_that_ran_not_the_ones_requested(context_factory):
    """`notch_frequencies` answers from the same resolver `notch` designs from, so a
    line dropped at Nyquist is visible in provenance instead of being invisible inside
    a (freq, harmonics) pair."""
    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 3,
                "notch_q": 30.0}))
    assert res.payload.provenance["notch_freqs"] == [60.0, 120.0, 180.0]


def test_a_notch_line_list_round_trips_through_provenance_as_a_list(context_factory):
    """A project file is JSON, so this key comes back as a list of numbers. It has to
    re-spell as a list of floats or a restored chain hashes differently from the live
    one under `detection_signal_key` and the cached trace is rebuilt on every drag."""
    from nanoscope.analysis.modules import (
        detection_signal_key, preprocess_params_from_provenance)

    lines = [60.0, 120.0, 180.0, 300.0, 420.0, 540.0]
    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "notch_enabled": True, "notch_freqs": lines, "notch_q": 30.0}))
    prov = dict(res.payload.provenance)
    prov["notch_freqs"] = list(prov["notch_freqs"])     # what JSON gives back
    back = preprocess_params_from_provenance(prov)
    assert back["notch_freqs"] == lines
    assert detection_signal_key(prov) == detection_signal_key(
        dict(prov, notch_freqs=list(lines)))

    # An explicit None (the series spelling) is DROPPED rather than left to shadow
    # notch_freq_hz -- `notch_freqs_param` reads None as "series", but a caller merging
    # this dict over form values would otherwise blank a real list.
    assert "notch_freqs" not in preprocess_params_from_provenance(
        dict(prov, notch_freqs=None))


# ---------------------------------------------------------------------------
# the NORMALIZE disclosure travels with the result
# ---------------------------------------------------------------------------
#
# `preprocess_for_detection` stage 7 rescales the whole trace before either detector
# sees it, and its own docstring states the consequence: "NORMALIZE IS LAST, AND IT
# TAKES THE PICOAMPS WITH IT ... depth_pA, the baseline, the thresholds and the
# current-derived pA/pA·s export fields retain legacy labels while carrying sigmas or span-fractions. The
# detector cannot detect that, so the disclosure is the panel's job."
#
# The panel did its job -- `DetectionPanel._refresh_normalize_units_note` banners for
# as long as the stage is on. But a banner does not TRAVEL: leave the panel, open
# Analysis > Plots, and the depth histogram of a normalized run drew with no caveat
# anywhere, on an axis labelled depth_pA. `AnalysisResult.warnings` is the channel
# that DOES travel -- the Summary panel, the Plots details pane (`_detection_notes`)
# and every export read it -- and it had no clause for the one stage that changes
# what the numbers MEAN rather than what they are.

def test_a_normalized_detection_says_so_in_its_own_warnings(context_factory):
    """The fact reaches every surface that prints AnalysisResult.warnings, without any
    of them knowing the Normalize stage exists."""
    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "normalize_enabled": True, "normalize_mode": "robust"}))
    hits = [w for w in res.warnings if "Normalize" in w]
    assert len(hits) == 1, res.warnings
    w = hits[0]
    assert "DIMENSIONLESS" in w
    # It has to name the columns by the names the user sees on the axis, or the
    # sentence is about an abstraction rather than about this table.
    assert "depth_pA" in w and "baseline_pA" in w
    assert "robust" in w
    # ...and it must NOT send the reader to frac_depth, which the affine rescale
    # destroys (measured 872x-1048x; see tests/core/test_signal_domain.py).
    assert "frac_depth" in w and "not reach for frac_depth" in w
    assert "every exported column" not in w.lower()
    assert "Index, time, source, review and status" in w


def test_an_ordinary_detection_stays_quiet_about_normalize(context_factory):
    """A guard that fires on everything trains the user to ignore it. The default
    chain does not normalize, so nothing is said."""
    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4}))
    assert not any("Normalize" in w for w in res.warnings)


def test_a_highpass_detection_separates_localization_from_measurement(context_factory):
    """High-pass finds bounds, while physical amplitudes come from the no-HP branch."""
    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "hp_enabled": True, "hp_cutoff_hz": 1.0}))
    hits = [w for w in res.warnings if "High-pass was ON" in w]
    assert len(hits) == 1
    assert "baseline_pA" in hits[0] and "frac_depth" in hits[0]
    assert "CANDIDATE LOCALIZATION ONLY" in hits[0]
    assert "measured" in hits[0] and "High-pass omitted" in hits[0]
    assert "start/end, dwell, rise/fall" in hits[0]
    assert "candidate locations only" not in hits[0]
    assert res.payload.provenance["characterization_hp_omitted"] is True


def test_both_detectors_disclose_it_because_both_run_the_stage(context_factory):
    """DetectSchmitt and DetectCusum both call preprocess_for_detection and both
    characterize on its output, so a disclosure on one of them is a disclosure that
    depends on which engine the user picked in a combo."""
    from nanoscope.analysis.modules import DetectCusum

    for module, extra in ((DetectSchmitt(), {"k_in": 5.0, "k_out": 2.0}),
                          (DetectCusum(), {"cusum_k": 0.75, "cusum_h": 28.0})):
        res = module.run(context_factory(
            make_event_trace(), fs=FS,
            params=dict(extra, min_dwell_s=5e-4, normalize_enabled=True,
                        normalize_mode="zscore")))
        assert any("Normalize" in w for w in res.warnings), module.id


def test_the_model_input_disclosure_names_the_clip_and_the_bound(context_factory):
    """That mode's damage is not merely a change of unit: it is BOUNDED, so two events
    of different real depth come out identical above the clip and no row carries a
    flag for it -- a depth histogram simply grows a spike at the edge that looks like
    a population. The panel banner splits on exactly this; so does this sentence."""
    from nanoscope.analysis.filters import MODEL_INPUT_MODE

    res = DetectSchmitt().run(context_factory(
        make_event_trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "normalize_enabled": True, "normalize_mode": MODEL_INPUT_MODE,
                "normalize_clip_sigma": 12.0}))
    w = next(x for x in res.warnings if "Normalize" in x)
    assert "CLIPPED" in w and "12" in w
    assert "monotone" in w
    assert "move threshold crossings or change the event count" in w


def test_the_disclosure_and_the_pooling_rule_name_the_stage_identically():
    """One vocabulary for one stage. The per-run sentence (here) and the cross-run
    pooling warning (core.sources.pool_comparability) both read
    `signal_domain_of`, so a user who meets both does not have to work out that they
    are about the same thing."""
    from nanoscope.analysis.modules import normalized_units_warning
    from nanoscope.core.sources import signal_domain_of

    prov = {"normalize_enabled": True, "normalize_mode": "minmax"}
    assert signal_domain_of(prov) in normalized_units_warning(prov)


def test_an_empty_normalized_detection_still_discloses():
    """Unlike the smoothing advisory -- which needs dwells to probe and is silent on an
    empty table because no depth is being reported -- this one fires with no rows at
    all: the run still reports a baseline and a threshold, in the wrong units."""
    from nanoscope.analysis.modules import _build_warnings

    empty = pd.DataFrame({"flag": pd.Series([], dtype="string")})
    warns = _build_warnings(
        {"normalize_enabled": True, "normalize_mode": "zscore"}, empty)
    assert any("Normalize" in w for w in warns)
