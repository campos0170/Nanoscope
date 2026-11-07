"""Numerical regressions from the November background-correction audit.

These assert scientific consequences, not just implementation structure: known
depth retention, an independent mask seeing erased events, endpoint residuals,
solver-domain refusal and warnings on the actual detection result.
"""
from pathlib import Path
import warnings

import numpy as np
import pytest

from conftest import make_context
from nanoscope.analysis import baseline as B
from nanoscope.analysis.modules import DetectSchmitt, preprocess_for_detection


def states(*, polarity=-1, dwell=.4, duration=12., seed=20261105):
    fs = 5000.
    t = np.arange(int(fs * duration)) / fs
    truth = 100 + 2 * np.sin(2 * np.pi * .07 * t)
    mask = np.zeros(t.size, bool)
    for start in np.arange(1, duration - .9, 2):
        mask |= (t >= start) & (t < start + dwell)
    x = truth - 40 * mask + np.random.default_rng(seed).normal(0, .25, t.size)
    sign = -polarity
    return sign * x, sign * truth, mask, fs


def quiet_result(x, fs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return B.baseline_wander_result(x, fs, **kwargs)


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("method", ["block_median", "masked_median", "robust_median", "savgol"])
def test_erased_long_events_are_measured_and_warned_for_every_runtime_method(method, polarity):
    x, truth, mask, fs = states(polarity=polarity)
    original = x.copy()
    with warnings.catch_warnings(), B.collect_baseline_warnings() as sink:
        warnings.simplefilter("ignore")  # the application's sink must still receive it
        r = B.baseline_wander_result(x, fs, method=method, window_s=.5, polarity=polarity)
    true_depth = np.median(polarity * (x[mask] - truth[mask]))
    remaining = np.median(polarity * (r.corrected[mask] - r.reference))
    assert remaining < .15 * true_depth
    check = r.health.check("detrend_attenuation")
    assert check.level == "warn" and check.value > .7
    assert any(isinstance(w, B.DetrendAttenuationWarning) for w in sink)
    assert any("independently identified states" in w for w in r.warnings)
    assert np.array_equal(x, original), "diagnostics/correction must preserve the raw input"


@pytest.mark.parametrize("polarity", [-1, 1])
def test_widened_window_control_preserves_depths_and_clears_attenuation_warning(polarity):
    x, truth, mask, fs = states(polarity=polarity)
    r = quiet_result(x, fs, method="masked_median", rough_s=2., base_s=.1, polarity=polarity)
    retained = np.median(polarity * (r.corrected[mask] - r.reference))
    true_depth = np.median(polarity * (x[mask] - truth[mask]))
    assert retained == pytest.approx(true_depth, rel=.005)
    assert r.health.check("detrend_attenuation").ok
    assert not r.warnings


@pytest.mark.parametrize("method", ["block_median", "percentile", "asls", "masked_median", "robust_median", "savgol"])
def test_short_events_and_mirror_control(method):
    x, truth, mask, fs = states(dwell=.004)
    a = quiet_result(x, fs, method=method, window_s=.5, polarity=-1)
    b = quiet_result(-x, fs, method=method, window_s=.5, polarity=1)
    assert np.median(a.reference - a.corrected[mask]) > 38
    np.testing.assert_allclose(a.baseline, -b.baseline, atol=1e-9, rtol=0)
    assert a.health.check("detrend_attenuation").ok


def test_one_long_state_among_many_short_ones_cannot_hide_behind_the_median():
    x, _, _, fs = states(dwell=.004, duration=20)
    # One long state amid numerous short events. Median intrusion is near zero.
    for start in np.arange(.2, 19.8, .15):
        i = int(start * fs)
        x[i:i+20] -= 40
    x[int(8*fs):int(8.4*fs)] -= 40
    r = quiet_result(x, fs, method="masked_median", polarity=-1)
    check = r.health.check("detrend_attenuation")
    assert check.value < .03
    assert not check.ok and "worst eligible state" in check.message


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("duty", [.4, .6, .8, .95])
def test_dense_states_cannot_receive_a_clean_health_verdict(duty, polarity):
    fs = 5000.
    t = np.arange(60000) / fs
    x = (100 - 40 * (t % 1 < duty)
         + np.random.default_rng(3).normal(0, .25, t.size)) * -polarity
    r = quiet_result(x, fs, method="masked_median", polarity=polarity)
    assert not r.health.ok
    assert r.warnings
    # At very high duty the independent majority-based mask itself is ambiguous;
    # retain the frame-free/reference health warnings instead of certifying it.
    if duty < .8:
        assert not r.health.check("detrend_attenuation").ok


def test_short_recording_says_attenuation_is_not_measured():
    x = 100 + np.random.default_rng(4).normal(0, .25, 5000)
    r = quiet_result(x, 5000, method="block_median", polarity=-1)
    c = r.health.check("detrend_attenuation")
    assert c.value is None
    assert "not measurable" in c.message and "does not establish" in c.message


def test_measured_warning_reaches_real_analysis_result_and_structured_provenance():
    x, _, _, fs = states()
    ctx = make_context(x.astype(np.float32), fs=fs, params={
        "detrend_enabled": True, "detrend_method": "masked_median",
        "polarity": -1, "min_dwell_s": .0001,
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = DetectSchmitt().run(ctx)
    assert result.payload.provenance["detrend_attenuation_risk"] is True
    assert any("Measured event/state attenuation risk" in w for w in result.warnings)
    assert any("Measured event/state attenuation risk" in w
               for w in result.payload.provenance["detrend_warnings"])


@pytest.mark.parametrize("method", ["block_median", "percentile", "asls"])
@pytest.mark.parametrize("reverse", [False, True])
def test_endpoint_is_bounded_and_cannot_extrapolate_a_false_plateau(method, reverse):
    x = np.r_[np.full(1500, 100.), np.full(500, 60.)]
    if reverse:
        x = x[::-1]
    b = B.estimate_baseline(x, 1000, method=method, window_s=.5, polarity=-1)
    # A plateau supplies a zero-width local data bound for these block statistics.
    if method != "asls":
        assert b.min() >= 60 and b.max() <= 100
        assert b[0] == x[0] and b[-1] == x[-1]


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("at_end", [False, True])
@pytest.mark.parametrize("method", ["block_median", "masked_median"])
def test_short_edge_events_survive_and_long_truncated_states_are_warned(method, at_end, polarity):
    fs = 5000.
    for dwell, should_warn in ((.004, False), (.4, True)):
        x = 100+np.random.default_rng(7).normal(0, .25, 60000)
        width = int(dwell*fs)
        event = slice(-width, None) if at_end else slice(0, width)
        x[event] -= 40
        x *= -polarity
        r = quiet_result(x, fs, method=method, window_s=.5, polarity=polarity)
        if should_warn:
            assert not r.health.check("detrend_attenuation").ok
            assert any("Measured event/state attenuation risk" in w for w in r.warnings)
        else:
            assert np.median(polarity*(r.corrected[event]-r.reference)) > 38


def test_edge_bounds_preserve_supported_interpolation_and_limit_extrapolation():
    centres = np.array([10., 30., 50.])
    knots = np.array([100., 90., 60.])
    bounds = ((98., 105.), (59., 61.))
    curve = B._expand(centres, knots, 70, edge_bounds=bounds)
    expected = B.PchipInterpolator(centres, knots)(np.arange(70))
    np.testing.assert_array_equal(curve[10:51], expected[10:51])
    assert np.all((curve[:10] >= 98) & (curve[:10] <= 105))
    assert np.all((curve[51:] >= 59) & (curve[51:] <= 61))


@pytest.mark.parametrize("polarity", [-1, 1])
def test_sloped_edges_remain_supported_instead_of_creating_a_false_step(polarity):
    fs = 100000.
    t = np.arange(300000)/fs
    truth = 100+45*np.sin(2*np.pi*.5*t)+30*t/t[-1]
    x = truth+np.random.default_rng(0).normal(0, 1.5, t.size)
    for a, b in ((30000,30300),(90000,90250),(150000,150400),(210000,210200),(270000,270350)):
        x[a:b] -= 30
    r = quiet_result(-polarity*x, fs, method="block_median", window_s=.3, polarity=polarity)
    # The edge half-window must follow the observed ramp, not invent a 20 pA step.
    edge = np.r_[np.arange(10000), np.arange(290000,300000)]
    assert np.max(np.abs(r.baseline[edge]-(-polarity*truth[edge]))) < 8
    from nanoscope.analysis.detect import detect_events
    df, _ = detect_events(r.corrected, fs, min_dwell_s=.0005)
    assert len(df) == 5


INVALID_ASLS = [
    ("asls_lam", -1), ("asls_lam", 0), ("asls_lam", np.nan), ("asls_lam", np.inf),
    ("asls_p", -.1), ("asls_p", 0), ("asls_p", 1), ("asls_p", 1.5), ("asls_p", np.nan),
    ("asls_niter", 0), ("asls_niter", -1), ("asls_niter", 1.5),
    ("asls_niter", np.nan), ("asls_niter", np.inf), ("asls_niter", True),
    ("asls_decim", -.1), ("asls_decim", 0), ("asls_decim", 1.1),
    ("asls_decim", np.inf), ("asls_decim", np.nan),
]


@pytest.mark.parametrize("key,value", INVALID_ASLS)
@pytest.mark.parametrize("runtime", [False, True])
def test_invalid_asls_parameters_refuse_before_a_result(key, value, runtime):
    x = np.linspace(100, 101, 5000)
    with pytest.raises(ValueError, match="asls_"):
        if runtime:
            preprocess_for_detection(x, 5000, {
                "detrend_enabled": True, "detrend_method": "asls",
                "detrend_" + key: value, "polarity": -1,
            })
        else:
            B.estimate_baseline(x, 5000, method="asls", polarity=-1, **{key: value})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["masked_median", "asls", "block_median"])
def test_nonfinite_gaps_are_refused_without_interpolation(method, bad):
    x, _, _, fs = states()
    x[100:110] = bad
    with pytest.raises(ValueError, match="finite"):
        B.baseline_wander_result(x, fs, method=method, polarity=-1)


def test_preview_and_runtime_share_one_estimate_and_constant_remains_noop(monkeypatch):
    x, _, _, fs = states(dwell=.004)
    calls = []
    estimate = B.estimate_baseline
    def record(*args, **kwargs):
        calls.append(kwargs)
        return estimate(*args, **kwargs)
    monkeypatch.setattr(B, "estimate_baseline", record)
    r = quiet_result(x, fs, method="masked_median", polarity=-1)
    assert len(calls) == 1
    np.testing.assert_array_equal(r.corrected, (x-r.baseline+r.reference).astype(np.float32))
    c = quiet_result(x, None, method="constant", polarity=-1)
    assert c.health is None and not c.warnings
    np.testing.assert_array_equal(c.corrected, x.astype(np.float32))


@pytest.mark.parametrize("method", B.METHODS)
def test_endpoint_policy_versions_only_the_changed_automatic_algorithms(method, monkeypatch):
    cfg = B.BaselineConfig(method=method).resolve(5000)
    current = cfg.to_provenance(5000)
    with monkeypatch.context() as historical:
        historical.setattr(B, "BLOCK_ENDPOINT_POLICY", None)
        old = cfg.to_provenance(5000)
    affected = method in ("block_median", "percentile", "asls")
    assert (current["baseline_fingerprint"] != old["baseline_fingerprint"]) == affected
    if affected:
        assert current["baseline_endpoint_policy"] == B.BLOCK_ENDPOINT_POLICY
        with pytest.raises(B.BaselinePolicyChangedError, match="Re-run detection"):
            B.BaselineConfig.reproducible_from(old)
        # Legacy controls remain restorable for an explicitly new calculation.
        assert B.BaselineConfig.from_provenance(old) == cfg
        with pytest.raises(ValueError, match="unsupported baseline endpoint policy"):
            B.BaselineConfig.from_provenance(current | {"baseline_endpoint_policy": "future-v99"})
        with pytest.raises(B.BaselinePolicyChangedError, match="Re-run detection"):
            B.BaselineConfig.reproducible_from(
                current | {"baseline_fingerprint": old["baseline_fingerprint"]})
    else:
        assert "baseline_endpoint_policy" not in current
    assert B.BaselineConfig.reproducible_from(current) == cfg


def test_runtime_provenance_reports_actual_endpoint_policy_and_clears_stale_policy():
    from nanoscope.analysis.modules import stamp_preprocess_provenance
    provenance = {}
    stamp_preprocess_provenance(provenance, {"detrend_enabled": True}, 5000, ())
    assert provenance["detrend_endpoint_policy"] == B.BLOCK_ENDPOINT_POLICY
    stamp_preprocess_provenance(provenance, {"detrend_enabled": True,
                                           "detrend_method": "masked_median"}, 5000, ())
    assert "detrend_endpoint_policy" not in provenance
    provenance["detrend_endpoint_policy"] = "old"
    stamp_preprocess_provenance(provenance, {"detrend_enabled": False}, 5000, ())
    assert "detrend_endpoint_policy" not in provenance


def test_obsolete_runtime_policy_cannot_replay_or_supply_a_cached_detection_signal():
    from nanoscope.analysis.modules import detection_signal, detection_signal_key
    cfg = B.DEFAULT_BASELINE.resolve(5000)
    old = cfg.to_provenance(5000) | {"detrend_enabled": True, "detrend_method": "block_median"}
    with pytest.raises(ValueError, match="Re-run detection"):
        B.BaselineConfig.reproducible_from(old)
    for fn in (lambda: detection_signal(np.arange(5000), 5000, old),
               lambda: detection_signal_key(old)):
        with pytest.raises(ValueError, match="Re-run detection"):
            fn()
    current = old | {"detrend_endpoint_policy": B.BLOCK_ENDPOINT_POLICY,
                     "detrend_sample_count_policy": B.SAMPLE_COUNT_POLICY}
    B.require_current_detrend_policy(current)
    assert B.BaselineConfig.reproducible_from(current) == cfg
    assert detection_signal_key(current)


@pytest.mark.parametrize("obsolete", ["runtime", "baseline"])
def test_obsolete_policy_refuses_direct_row_reconstruction_without_mutation(obsolete):
    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.events import EventSet
    from nanoscope.analysis.characterize import recharacterize_span
    from nanoscope.analysis.remeasure import remeasure_events
    from pandas.testing import assert_frame_equal
    x, _, _, fs = states(dwell=.004)
    cfg = B.BaselineConfig(method="block_median", window_s=.5)
    df, prov = detect_events(x, fs, polarity=-1, baseline_config=cfg, min_dwell_s=.0001)
    assert len(df) > 0
    old = dict(prov)
    if obsolete == "runtime":
        old.update(detrend_enabled=True, detrend_method="block_median")
        old.pop("detrend_endpoint_policy", None)
    else:
        old.pop("baseline_endpoint_policy", None)
    events = EventSet(df, old)
    before = events.df.copy(deep=True)
    first = before.iloc[0]
    with pytest.raises(B.BaselinePolicyChangedError, match="Re-run detection"):
        recharacterize_span(x, fs, int(first.start_idx), int(first.end_idx),
                            provenance=old, prior=first, polarity=-1)
    with pytest.raises(B.BaselinePolicyChangedError, match="Re-run detection"):
        remeasure_events(events, x, fs, config=B.DEFAULT_BASELINE)
    assert_frame_equal(events.df, before)
    assert events.provenance == old


def test_projection_validates_reference_but_allows_explicit_fresh_detection_of_old_recipe():
    from nanoscope.analysis.modules import detection_params_from_provenance
    x, _, _, fs = states(dwell=.004)
    old = B.DEFAULT_BASELINE.to_provenance(fs) | {
        "detrend_enabled": True, "detrend_method": "percentile", "detrend_window_s": .7,
        "percentile": 85., "polarity": -1, "min_dwell_s": .0001,
    }
    params = detection_params_from_provenance(old)
    assert params["detrend_window_s"] == .7 and params["detrend_percentile"] == 85
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = DetectSchmitt().run(make_context(x.astype(np.float32), fs=fs, params=params))
    assert fresh.payload.provenance["detrend_endpoint_policy"] == B.BLOCK_ENDPOINT_POLICY
    assert len(fresh.payload.df) > 0
    corrupt = old | {"baseline_reference_applied": True, "baseline_open_pore_reference": None}
    with pytest.raises(ValueError, match="lacks its recipe"):
        detection_params_from_provenance(corrupt)


@pytest.mark.corpus
def test_real_apo_recording_uses_same_estimate_and_reports_the_actual_health():
    path = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
    if not path.exists():
        pytest.skip("owner apo-ferritin recording is not available")
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document
    doc = open_document(path)
    ch = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT][0]
    x = np.asarray(doc.channels[ch.index], np.float64)
    fs = float(doc.meta.sample_rate_hz)
    # Bound runtime to the real segment containing the documented 37.13 s clog.
    x = x[int(30*fs):int(45*fs)]
    r = quiet_result(x, fs, method="masked_median", polarity=-1)
    expected = B.estimate_baseline(x, fs, method="masked_median", window_s=.5, polarity=-1)
    np.testing.assert_array_equal(r.baseline, expected)
    assert r.health.n_samples == x.size and np.isfinite(r.corrected).all()
    assert not r.health.check("detrend_attenuation").ok
    assert any("Measured event/state attenuation risk" in w for w in r.warnings)
