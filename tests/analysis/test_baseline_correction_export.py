"""The baseline-comparison export is a record, not a screenshot.

The two corrected columns can look alike at a glance while saying opposite things
about zero.  These tests hold the table, mode stamp, recipe, and all-or-nothing
two-file write contract together.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.conditioning import (
    BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
    BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
    baseline_correction_export_paths,
    baseline_correction_provenance,
    write_baseline_correction,
)


FS = 2_000.0


def _signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.array([101.0, 102.0, 98.0, 99.0], dtype=np.float32)
    fitted = np.array([100.0, 100.5, 100.5, 100.0], dtype=np.float32)
    zero = raw - fitted
    retained = raw - fitted + np.float32(np.median(fitted))
    return raw, fitted, zero, retained


def test_comparison_csv_and_sidecar_name_the_two_noninterchangeable_outputs(tmp_path):
    raw, fitted, zero, retained = _signals()
    visible = write_baseline_correction(
        tmp_path / "visible", raw, fitted, zero, FS,
        mode=BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
        units="pA", start_idx=40, recording="apo-700.abf", channel="current",
        channel_index=0, span="cursor selection", t0_s=3.25,
        baseline_params={"baseline_method": "masked_median", "baseline_base_s": 0.3},
        warnings=["reference fit is illustrative"],
    )
    runtime = write_baseline_correction(
        tmp_path / "runtime.csv", raw, fitted, retained, FS,
        mode=BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
        units="pA", start_idx=40,
        runtime_params={
            "detrend_enabled": True,
            "detrend_method": "masked_median",
            "detrend_window_s": 0.3,
        },
    )

    visible_df = pd.read_csv(visible["data"])
    assert list(visible_df.columns) == [
        "time_s", "raw_current_pA", "fitted_baseline_pA", "corrected_current_pA",
    ]
    assert np.array_equal(visible_df["raw_current_pA"].to_numpy(dtype=np.float32), raw)
    assert np.array_equal(visible_df["fitted_baseline_pA"].to_numpy(dtype=np.float32), fitted)
    assert np.array_equal(visible_df["corrected_current_pA"].to_numpy(dtype=np.float32), zero)
    assert visible_df["time_s"].iloc[0] == pytest.approx(40 / FS)

    visible_meta = json.loads(visible["meta"].read_text())
    runtime_meta = json.loads(runtime["meta"].read_text())
    assert visible_meta["correction_mode"] == BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY
    assert visible_meta["signal_basis"] == "baseline_display_residual"
    assert visible_meta["runtime_correction_active"] is False
    assert "display only" in visible_meta["corrected_signal_role"].lower()
    assert visible_meta["baseline_source"] == "open_pore_reference"
    assert visible_meta["comparison_scope"] == "open_pore_reference_display_only"
    assert visible_meta["excluded_runtime_stages"] == []
    assert visible_meta["warnings"] == ["reference fit is illustrative"]
    assert visible_meta["data_file"] == visible["data"].name

    assert runtime_meta["correction_mode"] == BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME
    assert runtime_meta["signal_basis"] == "baseline_runtime_corrected"
    assert runtime_meta["runtime_correction_active"] is True
    assert "retained" in runtime_meta["corrected_signal_role"].lower()
    assert runtime_meta["baseline_source"] == "baseline_drift_estimate"
    assert runtime_meta["comparison_scope"] == "baseline_drift_stage_only"
    assert runtime_meta["runtime_params"]["detrend_method"] == "masked_median"
    assert tuple(baseline_correction_export_paths(tmp_path / "visible")) == (
        visible["data"], visible["meta"],
    )


def test_sidecar_preserves_reference_recipe_and_only_baseline_drift_inputs():
    config = BaselineConfig(method="masked_median", rough_s=0.5, base_s=0.1)
    wide_form = {
        "detrend_enabled": True,
        "detrend_method": "masked_median",
        "detrend_window_s": 0.3,
        "detrend_rough_s": 1.0,
        "detrend_base_s": 0.3,
        "filter_cutoff_hz": 5_000.0,
        "median_enabled": True,
        "k_in": 5.0,
        "min_dwell_s": 1e-4,
        "baseline_rough_s": 2.0,
    }
    meta = baseline_correction_provenance(
        mode=BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
        fs=FS, n_samples=10, start_idx=0, units="pA",
        baseline_params=config, runtime_params=wide_form,
        cal_gain=10.0, cal_offset=-2.0,
    )

    assert meta["baseline_params"]["baseline_method"] == "masked_median"
    assert "baseline_fingerprint" in meta["baseline_params"]
    assert meta["runtime_params"]["detrend_method"] == "masked_median"
    assert "filter_cutoff_hz" not in meta["runtime_params"]
    assert "median_enabled" not in meta["runtime_params"]
    assert meta["comparison_scope"] == "baseline_drift_stage_only"
    assert meta["excluded_runtime_stages"] == ["median", "low_pass"]
    assert "k_in" not in meta["runtime_params"]
    assert "min_dwell_s" not in meta["runtime_params"]
    assert "baseline_rough_s" not in meta["runtime_params"]
    assert meta["cal_gain"] == 10.0 and meta["cal_offset"] == -2.0

    # Retained-offset output may be inspected before the user opts the stage into
    # detection.  The mode still names its numerical contract, while this flag says
    # whether the live chain was actually using it.
    inactive = baseline_correction_provenance(
        mode=BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
        fs=FS, n_samples=10, start_idx=0, units="pA",
        runtime_params={"detrend_enabled": False, "detrend_method": "block_median"},
    )
    assert inactive["runtime_correction_active"] is False


def test_bad_comparison_shapes_modes_and_samples_do_not_create_files(tmp_path):
    raw, fitted, zero, _retained = _signals()
    target = tmp_path / "bad.csv"
    with pytest.raises(ValueError, match="unknown baseline correction mode"):
        write_baseline_correction(target, raw, fitted, zero, FS, mode="looks_zeroish")
    with pytest.raises(ValueError, match="identical one-dimensional shapes"):
        write_baseline_correction(
            target, raw, fitted[:-1], zero, FS,
            mode=BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
        )
    with pytest.raises(ValueError, match="non-finite"):
        write_baseline_correction(
            target, raw, fitted, np.array([0.0, np.nan, 0.0, 0.0]), FS,
            mode=BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
        )
    with pytest.raises(ValueError, match="must not be stamped as a processing recipe"):
        write_baseline_correction(
            target, raw, fitted, zero, FS,
            mode=BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
            runtime_params={"detrend_enabled": True, "detrend_method": "masked_median"},
        )
    assert not any(path.exists() for path in baseline_correction_export_paths(target))


def test_failure_while_staging_the_sidecar_keeps_an_existing_pair_intact(tmp_path, monkeypatch):
    raw, fitted, zero, _retained = _signals()
    target = tmp_path / "comparison.csv"
    data_path, meta_path = baseline_correction_export_paths(target)
    data_path.write_text("OLD CSV\n", encoding="utf-8")
    meta_path.write_text('{"old": true}\n', encoding="utf-8")
    old_data, old_meta = data_path.read_bytes(), meta_path.read_bytes()

    def fail_sidecar_write(self: Path, *_args, **_kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(Path, "write_text", fail_sidecar_write)
    with pytest.raises(OSError, match="simulated full disk"):
        write_baseline_correction(
            target, raw, fitted, zero, FS,
            mode=BASELINE_CORRECTION_ZERO_CENTERED_DISPLAY,
        )

    assert data_path.read_bytes() == old_data
    assert meta_path.read_bytes() == old_meta


def test_a_nonfinite_runtime_recipe_cannot_leave_a_non_json_sidecar(tmp_path):
    raw, fitted, _zero, retained = _signals()
    target = tmp_path / "strict.json.csv"
    with pytest.raises(ValueError, match="Out of range float"):
        write_baseline_correction(
            target, raw, fitted, retained, FS,
            mode=BASELINE_CORRECTION_RETAINED_OFFSET_RUNTIME,
            runtime_params={
                "detrend_enabled": True,
                "detrend_method": "masked_median",
                "detrend_window_s": float("nan"),
            },
        )
    assert not any(path.exists() for path in baseline_correction_export_paths(target))
