"""Contradictory reference instructions cannot disappear in params projection."""
import pytest

from nanoscope.analysis.baseline import BaselineConfig, DEFAULT_BASELINE, baseline_config_from_params
from nanoscope.analysis.modules import detection_params_from_provenance


KEY = "baseline_open_pore_reference"
RECIPE = {"version": 1, "mode": "constant", "current_pA": -100.,
          "intervals_s": [], "interpolation": "constant", "notes": "known"}
REFERENCE = BaselineConfig(open_pore_reference=RECIPE)


@pytest.mark.parametrize("flag", ["baseline_reference_applied", "baseline_estimator_bypassed"])
@pytest.mark.parametrize("recipe", [{}, {KEY: None}])
@pytest.mark.parametrize("explicit", [{}, {"baseline_config": DEFAULT_BASELINE}, {"baseline_config": REFERENCE}])
def test_active_reference_without_flat_recipe_refuses_before_explicit_precedence(flag, recipe, explicit):
    settings = DEFAULT_BASELINE.to_provenance(50_000) | {flag: True} | recipe | explicit
    with pytest.raises(ValueError, match="lack.*recipe"):
        baseline_config_from_params(settings)


@pytest.mark.parametrize("explicit,flat", [
    (DEFAULT_BASELINE, RECIPE), (REFERENCE, None),
    (REFERENCE, RECIPE | {"current_pA": -110.}), (REFERENCE, {}),
])
def test_explicit_config_cannot_silently_shadow_a_different_reference(explicit, flat):
    with pytest.raises(ValueError):
        baseline_config_from_params({"baseline_config": explicit, KEY: flat})


@pytest.mark.parametrize("settings", [{}, {KEY: None},
    {KEY: None, "baseline_reference_applied": False, "baseline_estimator_bypassed": False}])
def test_ordinary_automatic_and_null_recipe_keep_default_identity(settings):
    config, source = baseline_config_from_params(settings)
    assert config is DEFAULT_BASELINE
    assert source == "default"
    assert config.to_provenance(50_000) == DEFAULT_BASELINE.to_provenance(50_000)


@pytest.mark.parametrize("settings", [
    {"baseline_config": REFERENCE}, {KEY: RECIPE},
    {"baseline_config": REFERENCE, KEY: RECIPE},
    {"baseline_config": REFERENCE, KEY: RECIPE, "baseline_reference_applied": True},
])
def test_valid_explicit_or_flat_recipes_remain_supported(settings):
    config, source = baseline_config_from_params(settings)
    assert config == REFERENCE and source == "user"


def test_explicit_automatic_and_flat_null_are_consistent():
    config, source = baseline_config_from_params({"baseline_config": DEFAULT_BASELINE, KEY: None})
    assert config is DEFAULT_BASELINE and source == "user"


@pytest.mark.parametrize("flag", ["baseline_reference_applied", "baseline_estimator_bypassed"])
@pytest.mark.parametrize("recipe", [{}, {KEY: None}])
def test_provenance_to_rerun_projection_refuses_missing_active_recipe(flag, recipe):
    provenance = DEFAULT_BASELINE.to_provenance(50_000) | {flag: True} | recipe
    with pytest.raises(ValueError):
        detection_params_from_provenance(provenance)


def test_valid_provenance_projection_keeps_reference_recipe():
    provenance = REFERENCE.to_provenance(50_000)
    assert detection_params_from_provenance(provenance)[KEY] == RECIPE
