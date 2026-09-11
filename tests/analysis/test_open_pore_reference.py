"""Explicit references are opt-in, reproducible, and never silently estimated."""
import hashlib
import json

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.open_pore_reference import (
    OPEN_PORE_REFERENCE_PARAM_KEY as KEY, OpenPoreReference, build_reference,
)


def constant(value=-100.0):
    return OpenPoreReference(mode="constant", current_pA=value)


def intervals(spans=((0.0, 0.01), (0.08, 0.1)), interpolation="linear"):
    return OpenPoreReference(mode="intervals", intervals_s=spans, interpolation=interpolation)


def test_disabled_config_keeps_legacy_identity_and_provenance():
    expected = {
        "baseline_method": "masked_median", "baseline_rough_s": .5,
        "baseline_base_s": .1, "baseline_decim": 66, "baseline_mask_sigma": 3.5,
        "baseline_dilate_ms": 1., "baseline_window_s": .5,
        "baseline_despike_size": 0, "baseline_fingerprint": "sha256:8e331d27a59a",
    }
    assert B.DEFAULT_BASELINE.to_provenance(50000) == expected
    assert B.BaselineConfig(open_pore_reference=None).to_provenance(50000) == expected
    assert KEY not in B.BASELINE_PARAM_KEYS
    cfg, source = B.baseline_config_from_params({KEY: None})
    assert cfg is B.DEFAULT_BASELINE and source == "default"
    parts = [f"{name}={getattr(cfg, name)!r}" for name in cfg._FIELDS]
    assert cfg.fingerprint() == "sha256:" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


@pytest.mark.parametrize("value", [-100., 0., 100.])
def test_supplied_current_preserves_sign_and_original_signal(value):
    signal = np.linspace(-20, 40, 100).astype(np.float32)
    original = signal.copy()
    base, details = build_reference(signal, 1000, constant(value))
    np.testing.assert_array_equal(base, np.full(100, value))
    np.testing.assert_array_equal(signal, original)
    assert base.dtype == np.float64
    assert details["user_assertion_not_verified"]
    assert not details["event_polarity_used_for_construction"]


def test_intervals_constant_is_pooled_sample_median_not_median_of_medians():
    x = np.r_[np.full(10, 10.), np.full(10, -999.), np.full(30, 20.)]
    recipe = intervals(((0., .01), (.02, .05)), "constant")
    base, details = build_reference(x, 1000, recipe)
    np.testing.assert_array_equal(base, np.full(x.size, 20.))
    assert details["interval_medians_pA"] == [10., 20.]
    assert details["pooled_median_pA"] == 20.
    assert details["selected_samples"] == 40


def test_linear_uses_only_robust_anchor_medians_and_discloses_endpoint_holds():
    x = np.full(100, -999.)
    x[:10] = np.arange(10.)
    x[0] = 1e4  # minority artifact cannot determine the interval median
    x[80:] = 20.
    base, details = build_reference(x, 1000, intervals())
    centres = [.0045, .0895]
    expected = np.interp(np.arange(100) / 1000, centres, [5.5, 20.])
    np.testing.assert_array_equal(base, expected)
    assert details["interval_centres_s"] == centres
    assert details["held_endpoint_fraction"] == .15
    assert details["largest_centre_gap_s"] == pytest.approx(.085)
    assert any("uncertainty" in s for s in details["assumptions"])
    x[20:70] = 1e7
    np.testing.assert_array_equal(build_reference(x, 1000, intervals())[0], base)


@pytest.mark.parametrize("recipe", [constant(), intervals(), intervals(interpolation="constant")])
def test_recipe_and_config_json_roundtrip_and_fingerprint(recipe):
    cfg = B.BaselineConfig(open_pore_reference=recipe.to_dict())
    prov = json.loads(json.dumps(cfg.to_provenance(1000), allow_nan=False))
    restored = B.BaselineConfig.from_provenance(prov)
    assert restored == cfg
    assert restored.fingerprint(1000) == prov["baseline_fingerprint"]
    assert restored.fingerprint(2000) != prov["baseline_fingerprint"]
    assert cfg.fingerprint(1000) != B.DEFAULT_BASELINE.fingerprint(1000)
    assert not cfg.is_masked
    assert cfg.resolve(1000) is cfg
    assert "user" in cfg.describe()
    assert B.BaselineConfig.reproducible_from(prov) == cfg
    assert B.baseline_config_from_params({KEY: recipe.to_dict()}) == (cfg, "user")


@pytest.mark.parametrize("bad", [
    None, [], {}, {"mode": "unknown"}, {"mode": "constant"},
    {"mode": "constant", "current_pA": True},
    {"mode": "constant", "current_pA": "100"},
    {"mode": "constant", "current_pA": float("nan")},
    {"mode": "constant", "current_pA": float("inf")},
    {"mode": "constant", "current_pA": 100, "unexpected": 1},
    {"mode": "constant", "current_pA": 100, "version": 2},
    {"mode": "constant", "current_pA": 100, "version": True},
    {"mode": "constant", "current_pA": 100, "interpolation": "linear"},
    {"mode": "constant", "current_pA": 100, "intervals_s": [[0, 1]]},
    {"mode": "intervals", "intervals_s": []},
    {"mode": "intervals", "intervals_s": [[0, 1]], "current_pA": 100},
    {"mode": "intervals", "intervals_s": [[0, 1]], "interpolation": "linear"},
    {"mode": "intervals", "intervals_s": [[0, 1]], "interpolation": "cubic"},
    {"mode": "intervals", "intervals_s": [[0, 1]], "notes": 123},
    {"mode": "intervals", "intervals_s": [[-1, 1]]},
    {"mode": "intervals", "intervals_s": [[1, 1]]},
    {"mode": "intervals", "intervals_s": [[1, 0]]},
    {"mode": "intervals", "intervals_s": [[0, 2], [1, 3]]},
    {"mode": "intervals", "intervals_s": [[2, 3], [0, 1]]},
    {"mode": "intervals", "intervals_s": [[0, float("nan")]]},
    {"mode": "intervals", "intervals_s": [[0, 1, 2]]},
])
def test_malformed_recipes_are_rejected(bad):
    with pytest.raises(ValueError):
        OpenPoreReference.from_dict(bad)


@pytest.mark.parametrize("spans", [((0, .004),), ((.099, .11),), ((.2, .3),)])
def test_recording_specific_bounds_and_minimum_samples(spans):
    with pytest.raises(ValueError):
        build_reference(np.ones(100), 1000, intervals(spans, "constant"))


@pytest.mark.parametrize("signal,fs", [([], 1000), ([1, np.nan], 1000),
    ([1, np.inf], 1000), ([[1, 2], [3, 4]], 1000), ([1], 0), ([1], -1), ([1], np.nan)])
def test_nonfinite_or_invalid_scientific_inputs_are_rejected(signal, fs):
    with pytest.raises(ValueError):
        build_reference(signal, fs, constant())


def test_explicit_door_bypasses_estimator_length_adaptation_and_polarity_guess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an automatic estimator must not construct the explicit reference")
    monkeypatch.setattr(B, "adapt_to_length", forbidden)
    monkeypatch.setattr(B, "_polarity_for", forbidden)
    monkeypatch.setattr(B, "derive_polarity", forbidden)
    monkeypatch.setattr(B, "baseline_health", forbidden)
    cfg = B.BaselineConfig(open_pore_reference=constant())
    result = B.characterisation_baseline(np.ones(2), 1000, config=cfg, cache=False)
    assert result.polarity == 0
    assert result.fallback is None
    assert result.health.ok is False
    assert result.provenance()["baseline_reference_applied"]
    assert result.provenance()["baseline_estimator_bypassed"]
    assert B.UNPROBED_DEFAULT_WARNING not in result.warnings
    without_health = B.estimate_baseline_result(np.ones(2), 1000, config=cfg, health=False)
    assert without_health.health is None and without_health.warnings
    np.testing.assert_array_equal(without_health.baseline, result.baseline)


@pytest.mark.parametrize("position", [0, 50, 99])
def test_rebuild_does_not_repair_user_reference_evidence(position):
    x = np.ones(100)
    x[position] = np.nan
    cfg = B.BaselineConfig(open_pore_reference=intervals())
    with pytest.raises(ValueError):
        B.characterisation_baseline(x, 1000, config=cfg, on_nonfinite="interpolate")


def test_reference_cache_identity_and_source_do_not_leak():
    B.clear_baseline_cache()
    x = np.ones(100)
    cfg = B.BaselineConfig(open_pore_reference=constant(100))
    a = B.characterisation_baseline(x, 1000, config=cfg, param_source="user")
    b = B.characterisation_baseline(x, 1000, config=cfg, param_source="restored")
    c = B.characterisation_baseline(x, 1000, config=cfg.replace(open_pore_reference=constant(120)))
    assert a.baseline is b.baseline and b.param_source == "restored"
    np.testing.assert_array_equal(c.baseline, np.full(100, 120))
    assert a.fingerprint != c.fingerprint


def test_malformed_active_provenance_never_falls_back():
    good = B.BaselineConfig(open_pore_reference=constant()).to_provenance(1000)
    for change in ({KEY: {}}, {"baseline_method": "unknown"}, {"baseline_fingerprint": None}):
        with pytest.raises(ValueError):
            B.BaselineConfig.reproducible_from(good | change)


@pytest.mark.parametrize("flag", ["baseline_reference_applied", "baseline_estimator_bypassed"])
@pytest.mark.parametrize("missing", [True, False])
def test_active_provenance_requires_recipe_but_explicit_none_can_disable(flag, missing):
    bad = {flag: True, "baseline_method": "masked_median", "baseline_fingerprint": "sha256:invalid"}
    if not missing:
        bad[KEY] = None
    for restore in (B.BaselineConfig.from_provenance, B.BaselineConfig.reproducible_from):
        with pytest.raises(ValueError, match="lacks its recipe"):
            restore(bad)
    assert B.baseline_config_from_params({KEY: None}) == (B.DEFAULT_BASELINE, "default")


@pytest.mark.parametrize("detector_name", ["schmitt", "cusum"])
def test_detectors_match_the_same_explicit_array_without_algorithm_change(detector_name):
    from nanoscope.analysis.detect import detect_events
    from nanoscope.analysis.detect_cusum import detect_events_cusum
    detector = detect_events if detector_name == "schmitt" else detect_events_cusum
    fs = 10000
    x = 100 + np.random.default_rng(197).normal(0, .5, 20000)
    x[3000:18000] -= 30
    cfg = B.BaselineConfig(open_pore_reference=constant(100))
    reference_df, reference_prov = detector(x, fs, baseline_config=cfg)
    explicit_df, explicit_prov = detector(x, fs, baseline=np.full(x.size, 100.))
    for column in ("start_idx", "end_idx", "depth_pA", "baseline_pA", "ecd_pA_s", "snr", "accepted"):
        np.testing.assert_array_equal(reference_df[column], explicit_df[column])
    assert reference_prov["polarity"] == explicit_prov["polarity"] == -1
    assert reference_prov["noise_rms_pA"] == explicit_prov["noise_rms_pA"]
    assert reference_prov[KEY] == constant(100).to_dict()
