"""Exported endpoint semantics cannot be mistaken for legacy extrapolation."""
import json

import numpy as np
import pytest

from nanoscope.analysis.baseline import BaselineConfig, BLOCK_ENDPOINT_POLICY
from nanoscope.analysis.conditioning import (
    BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME as RETAINED,
    BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY as ZERO,
    baseline_correction_provenance,
    conditioned_trace,
    conditioning_params,
    processed_trace_provenance,
    write_baseline_correction,
    write_processed_trace,
)


COMMON = dict(fs=1000.0, n_samples=100, start_idx=0, units="pA")


@pytest.mark.parametrize("method", ["block_median", "percentile", "asls"])
def test_processed_export_stamps_actual_endpoint_policy_not_a_caller_claim(method):
    params = {"detrend_enabled": True, "detrend_method": method,
              "detrend_endpoint_policy": "legacy-or-fabricated"}
    meta = processed_trace_provenance(params=params, **COMMON)
    assert meta["detrend_endpoint_policy"] == BLOCK_ENDPOINT_POLICY
    assert "detrend_endpoint_policy" not in meta["params"]
    assert "detrend_endpoint_policy" not in conditioning_params(params)
    assert params["detrend_endpoint_policy"] == "legacy-or-fabricated"


@pytest.mark.parametrize("method", ["block_median", "percentile", "asls"])
@pytest.mark.parametrize("enabled", [False, True])
def test_comparison_export_records_calculated_stage_even_before_runtime_opt_in(method, enabled):
    meta = baseline_correction_provenance(mode=RETAINED, runtime_params={
        "detrend_method": method, "detrend_enabled": enabled,
        "detrend_endpoint_policy": "fabricated"}, **COMMON)
    assert meta["runtime_correction_active"] is enabled
    assert meta["detrend_endpoint_policy"] == BLOCK_ENDPOINT_POLICY
    assert "detrend_endpoint_policy" not in meta["runtime_params"]


@pytest.mark.parametrize("method", ["constant", "masked_median", "robust_median", "savgol"])
def test_unaffected_methods_do_not_claim_block_endpoint_policy(method):
    params = {"detrend_method": method, "detrend_enabled": True}
    assert "detrend_endpoint_policy" not in processed_trace_provenance(params=params, **COMMON)
    assert "detrend_endpoint_policy" not in baseline_correction_provenance(
        mode=RETAINED, runtime_params=params, **COMMON)


def test_disabled_processed_stage_unknown_comparison_and_open_pore_reference_are_distinct():
    meta = processed_trace_provenance(params={"detrend_enabled": False,
                                              "detrend_method": "block_median"}, **COMMON)
    assert "detrend_endpoint_policy" not in meta
    assert "detrend_endpoint_policy" not in baseline_correction_provenance(mode=RETAINED, **COMMON)
    meta = baseline_correction_provenance(mode=ZERO,
        baseline_params=BaselineConfig(method="block_median"), **COMMON)
    assert "detrend_endpoint_policy" not in meta
    assert meta["baseline_params"]["baseline_endpoint_policy"] == BLOCK_ENDPOINT_POLICY


def test_policy_reaches_written_sidecars_and_processed_samples_reproduce(tmp_path):
    raw = np.linspace(80, 110, 1000, dtype=np.float32)
    params = {"detrend_enabled": True, "detrend_method": "block_median",
              "detrend_window_s": .1, "filter_cutoff_hz": None}
    corrected = conditioned_trace(raw, 1000, params)
    output = write_processed_trace(tmp_path / "processed.csv", corrected, 1000,
                                   params=params, units="pA")
    meta = json.loads(output["meta"].read_text())
    assert meta["detrend_endpoint_policy"] == BLOCK_ENDPOINT_POLICY
    np.testing.assert_array_equal(conditioned_trace(raw, meta["sample_rate_hz"], meta["params"]), corrected)
    # The numeric columns still round-trip; stamping changes only provenance.
    np.testing.assert_array_equal(np.loadtxt(output["data"], delimiter=",", skiprows=1)[:, 1].astype(np.float32), corrected)
    output = write_baseline_correction(tmp_path / "comparison.csv", raw, raw,
        np.zeros_like(raw), 1000, mode=RETAINED, runtime_params=params, units="pA")
    assert json.loads(output["meta"].read_text())["detrend_endpoint_policy"] == BLOCK_ENDPOINT_POLICY
