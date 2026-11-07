"""Local-noise rail replay must show the comparison the detector actually used."""
import copy

import numpy as np
import pytest

from nanoscope.analysis import detect
from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.detection_thresholds import (
    RAIL_ENTER, RAIL_EXIT, ThresholdsUnavailable, detection_threshold_tracks,
    threshold_refusal, threshold_signal_key,
)
from nanoscope.analysis.open_pore_reference import OpenPoreReference

FS = 10_000.0
N = 100_000


def _record(policy="local", polarity=-1, dtype=np.float64, filter_cutoff_hz=None):
    rng = np.random.default_rng(75661)
    sigma = np.ones(N)
    sigma[40_000:60_000] = 8.0
    raw = 100.0 + rng.normal(size=N) * sigma
    raw[10_000:10_200] += polarity * 30
    raw[80_000:80_200] += polarity * 30
    raw = raw.astype(dtype)
    cfg = BaselineConfig(open_pore_reference=OpenPoreReference(mode="constant", current_pA=100))
    df, prov = detect.detect_events(
        raw, FS, polarity=polarity, baseline_config=cfg, noise_policy=policy,
        min_dwell_s=0.0005, filter_cutoff_hz=filter_cutoff_hz)
    return raw, df, prov


@pytest.mark.parametrize("polarity", [-1, 1])
@pytest.mark.parametrize("policy", ["local", "auto"])
def test_rails_reproduce_the_actual_normalized_detector_comparison(monkeypatch, polarity, policy):
    original = detect._hysteresis_runs
    comparison = {}

    def capture(values, n, enter, leave):
        comparison.update(values=values[:n].copy(), enter=enter, leave=leave)
        return original(values, n, enter, leave)

    monkeypatch.setattr(detect, "_hysteresis_runs", capture)
    raw, df, prov = _record(policy, polarity)
    assert prov["noise_policy_effective"] == "local"
    tracks = detection_threshold_tracks(raw, FS, prov, module_id="detect.schmitt", max_points=0)
    by = {r.label: r for r in tracks.rails}
    # Derive the actual scale from the values handed to the threshold kernel,
    # independently of calling the profile helper again as the replay does.
    scale = polarity * (raw - 100) / comparison["values"]
    for label, multiplier in ((RAIL_ENTER, comparison["enter"]),
                              (RAIL_EXIT, comparison["leave"])):
        np.testing.assert_array_equal(by[label].idx, np.arange(N))
        np.testing.assert_allclose(by[label].values, 100 + polarity * multiplier * scale,
                                   rtol=1e-13, atol=1e-12)
    assert len(df) >= 2
    assert "local sigma" in tracks.description
    assert "SNR uses its global noise" in tracks.description


def test_decimation_keeps_noise_steps_when_the_baseline_is_flat():
    raw, _df, prov = _record()
    tracks = detection_threshold_tracks(raw, FS, prov, max_points=120)
    ends = np.asarray(prov["noise_profile_tile_ends"])
    scales = np.asarray(prov["noise_profile_scales_pA"])
    step = int(ends[np.argmax(np.abs(np.diff(scales)))])
    for rail in tracks.rails:
        # The largest local-sigma change has a knot on both sides. Decimating
        # the constant baseline first would lose it entirely.
        assert step - 1 in rail.idx and step in rail.idx
        assert len(rail.idx) <= 122


def test_float32_cached_filter_signal_replays_without_false_digest_refusal():
    from nanoscope.analysis.modules import detection_signal
    raw, _df, prov = _record(filter_cutoff_hz=1500)
    cached = detection_signal(raw, FS, prov).astype(np.float32)
    tracks = detection_threshold_tracks(cached, FS, prov)
    assert len(tracks.rails) == 2 and "local sigma" in tracks.description


def test_legacy_missing_noise_policy_keeps_global_comparison():
    raw, _df, prov = _record(policy="global")
    legacy = {key: value for key, value in prov.items()
              if not key.startswith(("noise_policy", "noise_window", "noise_local",
                                     "noise_profile", "noise_algorithm"))}
    tracks = detection_threshold_tracks(raw, FS, legacy, max_points=0)
    for rail, key in zip(tracks.rails, ("threshold_in_pA", "threshold_out_pA")):
        np.testing.assert_allclose(rail.values, 100 - prov[key], rtol=0, atol=0)
    assert "local sigma" not in tracks.description


@pytest.mark.parametrize("key,value", [
    ("noise_algorithm_version", 999), ("noise_policy", "unknown"),
    ("noise_policy_effective", "unknown"), ("noise_window_s", float("nan")),
    ("noise_window_s", -0.1), ("noise_window_s", None),
])
def test_incomplete_or_unknown_noise_recipe_refuses_before_replay(key, value):
    raw, _df, prov = _record()
    bad = {**prov, key: value}
    assert threshold_refusal("detect.schmitt", bad) is not None
    with pytest.raises(ThresholdsUnavailable):
        detection_threshold_tracks(raw, FS, bad)


@pytest.mark.parametrize("key", ["noise_algorithm_version", "noise_window_s", "noise_policy"])
def test_missing_local_noise_recipe_field_refuses(key):
    raw, _df, prov = _record()
    prov.pop(key)
    with pytest.raises(ThresholdsUnavailable):
        detection_threshold_tracks(raw, FS, prov)


@pytest.mark.parametrize("target", ["noise_profile_sha256", "noise_profile_tile_ends",
                                    "noise_profile_scales_pA", "noise_window_samples",
                                    "noise_local_scale_min_pA", "noise_local_scale_median_pA",
                                    "noise_local_scale_max_pA"])
def test_tampered_local_profile_is_never_drawn(target):
    raw, _df, prov = _record()
    bad = copy.deepcopy(prov)
    if target == "noise_profile_sha256":
        bad[target] = "0" * 64
    elif isinstance(bad[target], list):
        bad[target][len(bad[target]) // 2] *= 1.1
    else:
        bad[target] *= 1.1
    with pytest.raises(ThresholdsUnavailable, match="record|reproduce"):
        detection_threshold_tracks(raw, FS, bad)


@pytest.mark.parametrize("key,value", [
    ("noise_policy", "global"), ("noise_policy_effective", "global"),
    ("noise_algorithm_version", 2), ("noise_window_s", 0.1),
    ("noise_window_samples", 500), ("noise_profile_sha256", "different"),
    ("noise_profile_scales_pA", [1, 2]), ("noise_profile_tile_ends", [2500, 5000]),
])
def test_threshold_cache_invalidates_for_noise_contract_changes(key, value):
    _raw, _df, prov = _record()
    assert threshold_signal_key(prov) != threshold_signal_key({**prov, key: value})
    hash(threshold_signal_key(prov))


@pytest.mark.parametrize("key,value", [
    ("noise_algorithm_version", True), ("noise_algorithm_version", 1.0),
    ("noise_window_s", True), ("noise_window_s", "0.25"),
    ("k_in", np.nan), ("k_in", True), ("k_in", "5"), ("k_in", 10 ** 500),
    ("k_out", 0), ("k_out", 6),
])
def test_noise_recipe_rejects_wrong_types_and_invalid_multipliers(key, value):
    raw, _df, prov = _record()
    bad = {**prov, key: value}
    assert threshold_refusal("detect.schmitt", bad) is not None
    with pytest.raises(ThresholdsUnavailable):
        detection_threshold_tracks(raw, FS, bad)


@pytest.mark.parametrize("mutation", [
    "fractional_end", "float_ends", "zero_end", "duplicate_end", "out_of_bounds_end",
    "too_many_ends", "nonfinite_scale", "negative_scale", "boolean_scales", "float_width",
    "changed_enter_multiplier", "changed_exit_multiplier", "missing_range", "wrong_range",
    "fictitious_scalar_floor", "nonfinite_range",
])
def test_local_profile_rejects_malformed_boundaries_scales_and_contradictory_ranges(mutation):
    raw, _df, original = _record()
    prov = copy.deepcopy(original)
    if mutation == "fractional_end":
        prov["noise_profile_tile_ends"][0] += .1  # must not silently truncate before digest verification
    elif mutation == "float_ends":
        prov["noise_profile_tile_ends"] = [float(v) for v in prov["noise_profile_tile_ends"]]
    elif mutation == "zero_end":
        prov["noise_profile_tile_ends"][0] = 0
    elif mutation == "duplicate_end":
        prov["noise_profile_tile_ends"][1] = prov["noise_profile_tile_ends"][0]
    elif mutation == "out_of_bounds_end":
        prov["noise_profile_tile_ends"][-1] += 1
    elif mutation == "too_many_ends":
        prov["noise_profile_tile_ends"] = list(range(1, 4098))
        prov["noise_profile_scales_pA"] = [1.] * 4097
    elif mutation in ("nonfinite_scale", "negative_scale"):
        prov["noise_profile_scales_pA"][0] = np.nan if mutation == "nonfinite_scale" else -1.
    elif mutation == "boolean_scales":
        prov["noise_profile_scales_pA"] = [True] * len(prov["noise_profile_tile_ends"])
    elif mutation == "float_width":
        prov["noise_window_samples"] += .1
    elif mutation == "changed_enter_multiplier":
        prov["k_in"] *= 1.2
    elif mutation == "changed_exit_multiplier":
        prov["k_out"] *= 1.2
    elif mutation == "missing_range":
        prov.pop("threshold_out_range_pA")
    elif mutation == "wrong_range":
        prov["threshold_in_range_pA"][0] *= 1.1
    elif mutation == "fictitious_scalar_floor":
        prov["threshold_out_pA"] = prov["threshold_out_range_pA"][0]
    elif mutation == "nonfinite_range":
        prov["threshold_out_range_pA"][1] = np.inf
    with pytest.raises(ThresholdsUnavailable):
        detection_threshold_tracks(raw, FS, prov)


def test_local_profile_with_missing_policy_and_effective_policy_is_not_legacy_global():
    raw, _df, prov = _record()
    prov.pop("noise_policy")
    prov.pop("noise_policy_effective")
    assert "incomplete" in threshold_refusal("detect.schmitt", prov)
    with pytest.raises(ThresholdsUnavailable, match="incomplete"):
        detection_threshold_tracks(raw, FS, prov)


@pytest.mark.parametrize("key", ["noise_window_max_samples", "noise_local_tiles"])
@pytest.mark.parametrize("mutation", ["wrong_value", "float_value"])
def test_profile_diagnostic_geometry_must_match_recorded_tile_boundaries(key, mutation):
    raw, _df, prov = _record()
    prov[key] = prov[key] + 1 if mutation == "wrong_value" else float(prov[key])
    with pytest.raises(ThresholdsUnavailable, match="recorded local noise profile"):
        detection_threshold_tracks(raw, FS, prov)
