"""A transient restore conflict remains a refusal at every baseline entry point."""
import pytest

from nanoscope.analysis.baseline import BaselineConfig, DEFAULT_BASELINE, baseline_config_from_params
from nanoscope.analysis.modules import detection_params_from_provenance
from nanoscope.analysis.reference_context import validate_open_pore_context


ERROR = "baseline_reference_restore_error"
MESSAGE = "The saved parameters and measurement provenance contain contradictory open-pore references."
RECIPE = {"version": 1, "mode": "constant", "current_pA": -100.,
          "intervals_s": [], "interpolation": "constant"}
REFERENCE = BaselineConfig(open_pore_reference=RECIPE)


@pytest.mark.parametrize("settings", [
    {}, DEFAULT_BASELINE.to_provenance(), REFERENCE.to_provenance(50_000.),
    {"baseline_config": DEFAULT_BASELINE}, {"baseline_config": REFERENCE},
    {"baseline_open_pore_reference": None},
])
@pytest.mark.parametrize("resolver", [baseline_config_from_params, BaselineConfig.from_provenance,
                                     BaselineConfig.reproducible_from, detection_params_from_provenance])
def test_explicit_restore_error_refuses_even_without_active_flags(settings, resolver):
    with pytest.raises(ValueError, match="contradictory open-pore"):
        resolver(dict(settings, **{ERROR: MESSAGE}))


def test_context_gate_refuses_transient_error_before_automatic_fastpath():
    with pytest.raises(ValueError, match="contradictory open-pore"):
        validate_open_pore_context({ERROR: MESSAGE}, units="pA", module_id="detect.schmitt")


@pytest.mark.parametrize("error", [None, ""])
def test_inactive_error_marker_does_not_change_default_identity(error):
    assert baseline_config_from_params({ERROR: error}) == (DEFAULT_BASELINE, "default")
