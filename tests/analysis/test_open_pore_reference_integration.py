"""Independent production-path checks for recording-specific user references.

These tests do not validate automatic reference inference or adaptive noise: the
new feature asserts user evidence while leaving each existing detector intact.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import BATCH_SCOPE_KEY, DEFAULT_BASELINE, BaselineConfig
from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt, detection_params_from_provenance
from nanoscope.analysis.open_pore_reference import build_reference
from nanoscope.analysis.reference_context import validate_open_pore_context
from nanoscope.analysis.remeasure import remeasure_events
from nanoscope.core.project import ProjectState, ResultRec, load_project, save_project


KEY = "baseline_open_pore_reference"
FS = 10_000.0
DETECTORS = [(DetectSchmitt, detect_events), (DetectCusum, detect_events_cusum)]
MEASUREMENTS = ["start_idx", "end_idx", "baseline_pA", "depth_pA", "ecd_pA_s", "snr", "accepted", "flag"]


def constant(value=100.0):
    return {"version": 1, "mode": "constant", "current_pA": float(value),
            "intervals_s": [], "interpolation": "constant", "notes": "independent reference"}


def regions(interpolation="linear"):
    return {"version": 1, "mode": "intervals", "current_pA": None,
            "intervals_s": [[.1, .2], [1.8, 1.9]],
            "interpolation": interpolation, "notes": "selected open-pore intervals"}


def signal(level=100.0, majority=False):
    x = level + np.random.default_rng(181).normal(0., .3, 20_000)
    direction = -1 if level > 0 else 1
    if majority:
        x[3000:18000] += direction * 30
    else:
        for a, b in ((3000, 3200), (8000, 8300), (15000, 15200)):
            x[a:b] += direction * 30
    return x.astype(np.float32)


def params(**changes):
    return {"polarity": None, "filter_cutoff_hz": None,
            "min_dwell_s": .001, "merge_gap_s": None, **changes}


def run(context_factory, module, x, recipe, **changes):
    settings = params(**{KEY: recipe, **changes})
    ctx = context_factory(x, fs=FS, units="pA", path="reference-integration.h5", params=settings)
    return module().run(ctx)


@pytest.mark.parametrize("module,kernel", DETECTORS)
@pytest.mark.parametrize("level", [-100., 100.])
def test_production_supplied_reference_keeps_signed_current_and_recalculates_auto(
    context_factory, module, kernel, level,
):
    x = signal(level, majority=True)
    untouched = x.copy()
    result = run(context_factory, module, x, constant(level))
    expected, prov = kernel(x, FS, baseline=np.full(x.size, level), **params())
    assert result.payload.provenance["polarity"] == prov["polarity"] == (-1 if level > 0 else 1)
    assert result.payload.provenance["polarity_requested"] is None
    assert len(result.payload.df) > 0
    for column in MEASUREMENTS:
        np.testing.assert_array_equal(result.payload.df[column], expected[column])
    assert result.payload.provenance["noise_rms_pA"] == prov["noise_rms_pA"]
    assert result.payload.provenance[KEY] == constant(level)
    assert result.payload.provenance["baseline_reference_applied"] is True
    assert result.payload.provenance["baseline_estimator_bypassed"] is True
    assert any("not verified" in warning for warning in result.warnings)
    np.testing.assert_array_equal(x, untouched)


@pytest.mark.parametrize("module,kernel", DETECTORS)
@pytest.mark.parametrize("interpolation", ["constant", "linear"])
def test_selected_regions_production_path_matches_explicit_numerical_reference(
    context_factory, module, kernel, interpolation,
):
    x = signal() + np.linspace(0., 10., 20_000, dtype=np.float32)
    recipe = regions(interpolation)
    result = run(context_factory, module, x, recipe)
    baseline, _ = build_reference(x, FS, recipe)
    expected, expected_prov = kernel(x, FS, baseline=baseline, **params())
    for column in MEASUREMENTS:
        np.testing.assert_array_equal(result.payload.df[column], expected[column])
    assert result.payload.provenance["polarity"] == expected_prov["polarity"] == -1
    assert result.payload.provenance[KEY] == recipe


@pytest.mark.parametrize("module,kernel", DETECTORS)
def test_explicit_reference_does_not_override_manual_polarity(context_factory, module, kernel):
    result = run(context_factory, module, signal(), constant(), polarity=1)
    assert result.payload.provenance["polarity"] == 1
    assert result.payload.provenance["polarity_requested"] == 1
    assert result.payload.provenance["polarity_source"] == "provided"


@pytest.mark.parametrize("module,kernel", DETECTORS)
def test_automatic_absent_null_and_explicit_default_are_numerically_identical(context_factory, module, kernel):
    x = signal()
    results = []
    for extra in ({}, {KEY: None}, {"baseline_config": DEFAULT_BASELINE}):
        ctx = context_factory(x, fs=FS, units="pA", path="same-recording.h5", params=params(**extra))
        results.append(module().run(ctx))
    for other in results[1:]:
        pd.testing.assert_frame_equal(results[0].payload.df, other.payload.df, check_exact=True)
        for name in ("baseline_fingerprint", "noise_rms_pA", "polarity"):
            assert other.payload.provenance[name] == results[0].payload.provenance[name]
        assert KEY not in other.payload.provenance


@pytest.mark.parametrize("module,kernel", DETECTORS)
@pytest.mark.parametrize("unit", ["nA", "A", "mV", "", None])
def test_module_refuses_unconverted_or_unknown_units(context_factory, module, kernel, unit):
    ctx = context_factory(signal(), fs=FS, units=unit, params=params(**{KEY: constant()}))
    with pytest.raises(ValueError, match="pA"):
        module().run(ctx)


@pytest.mark.parametrize("module,kernel", DETECTORS)
@pytest.mark.parametrize("stage", ["detrend_enabled", "hp_enabled", "normalize_enabled"])
def test_module_refuses_nonphysical_reference_domain(context_factory, module, kernel, stage):
    with pytest.raises(ValueError, match="cannot be combined"):
        run(context_factory, module, signal(), constant(), **{stage: True})


@pytest.mark.parametrize("module,kernel", DETECTORS)
@pytest.mark.parametrize("scope", [{BATCH_SCOPE_KEY: ("other", 0)},
                                  {BATCH_SCOPE_KEY: None}, {"baseline_param_source": "batch"}])
def test_module_never_copies_explicit_reference_through_batch(context_factory, module, kernel, scope):
    with pytest.raises(ValueError, match="one recording"):
        run(context_factory, module, signal(), constant(), **scope)


@pytest.mark.parametrize("module_id", ["detect.keras", "detect.reconciled", None])
def test_context_refuses_unvalidated_detector_families(module_id):
    with pytest.raises(ValueError, match="Schmitt and CUSUM"):
        validate_open_pore_context({KEY: constant()}, units="pA", module_id=module_id)


def test_context_no_reference_preserves_existing_unrestricted_workflows():
    validate_open_pore_context({"hp_enabled": True, "normalize_enabled": True},
                              units="nA", module_id="detect.keras", batch=True)


@pytest.mark.parametrize("recipe", [constant(), regions()])
def test_project_roundtrip_reproduces_production_measurements_and_auto(context_factory, tmp_path, recipe):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, recipe)
    state = ProjectState(results=[ResultRec(
        "reference-integration.h5", 0, result.module_id, result.payload.df,
        result.payload.provenance, params=dict(result.params),
        calibration_fingerprint=result.calibration_fingerprint,
        baseline_fingerprint=result.baseline_fingerprint, warnings=result.warnings,
    )])
    restored = load_project(save_project(tmp_path / "reference.nanoscope", state)).results[0]
    assert restored.params[KEY] == restored.provenance[KEY] == recipe
    assert restored.baseline_fingerprint == result.baseline_fingerprint
    cfg = BaselineConfig.reproducible_from(restored.provenance)
    assert cfg.open_pore_reference.to_dict() == recipe
    from_provenance = detection_params_from_provenance(restored.provenance)
    assert from_provenance[KEY] == recipe
    for settings in (restored.params, from_provenance):
        ctx = context_factory(x, fs=FS, units="pA", path="reference-integration.h5", params=settings)
        rerun = DetectSchmitt().run(ctx)
        for column in MEASUREMENTS:
            np.testing.assert_array_equal(rerun.payload.df[column], result.payload.df[column])
        assert rerun.payload.provenance["polarity_requested"] is None
    assert any("not verified" in warning for warning in restored.warnings)


@pytest.mark.parametrize("old,new", [(None, constant()), (constant(), constant(110)), (constant(), None)])
def test_changing_reference_cannot_masquerade_as_remeasurement(context_factory, old, new):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, old)
    before = result.payload.df.copy(deep=True)
    cfg = BaselineConfig(open_pore_reference=new)
    with pytest.raises(ValueError, match="fresh detection"):
        remeasure_events(result.payload, x, FS, config=cfg)
    pd.testing.assert_frame_equal(result.payload.df, before, check_exact=True)


@pytest.mark.parametrize("recipe", [constant(), regions()])
def test_same_reference_remeasurement_and_event_rebuild_preserve_reference(context_factory, recipe):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, recipe)
    cfg = BaselineConfig.reproducible_from(json.loads(json.dumps(result.payload.provenance)))
    remeasured = remeasure_events(result.payload, x, FS, config=cfg)
    for column in ("event_id", "start_idx", "end_idx", "baseline_pA", "depth_pA", "ecd_pA_s"):
        np.testing.assert_array_equal(remeasured.events.df[column], result.payload.df[column])
    prior = result.payload.df.iloc[[0]]
    row = recharacterize_span(x, FS, int(prior.iloc[0].start_idx), int(prior.iloc[0].end_idx),
                              provenance=result.payload.provenance, prior=prior)
    assert row.attrs["baseline_source"] == "recording"
    assert float(row.iloc[0].baseline_pA) == float(prior.iloc[0].baseline_pA)
    assert float(row.iloc[0].depth_pA) == float(prior.iloc[0].depth_pA)


def test_cropped_recording_cannot_silently_rebuild_from_flank_median(context_factory):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, regions())
    prior = result.payload.df.iloc[[0]]
    with pytest.raises(ValueError, match="outside this recording"):
        recharacterize_span(x[:10_000], FS, int(prior.iloc[0].start_idx), int(prior.iloc[0].end_idx),
                            provenance=result.payload.provenance, prior=prior)


@pytest.mark.parametrize("invalid", [{"normalize_enabled": True}, {"hp_enabled": True},
                                     {"detrend_enabled": True}, {"units": "nA"}])
def test_rebuild_refuses_malformed_reference_signal_domain(context_factory, invalid):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, constant())
    prior = result.payload.df.iloc[[0]]
    provenance = dict(result.payload.provenance) | invalid
    with pytest.raises(ValueError):
        recharacterize_span(x, FS, int(prior.iloc[0].start_idx), int(prior.iloc[0].end_idx),
                            provenance=provenance, prior=prior)


@pytest.mark.parametrize("remove", [False, True])
def test_rebuild_cannot_substitute_automatic_baseline_for_missing_active_recipe(context_factory, remove):
    x = signal()
    result = run(context_factory, DetectSchmitt, x, constant())
    prior = result.payload.df.iloc[[0]]
    provenance = dict(result.payload.provenance)
    assert provenance["baseline_reference_applied"]
    if remove:
        provenance.pop(KEY)
    else:
        provenance[KEY] = None
    with pytest.raises(ValueError):
        recharacterize_span(x, FS, int(prior.iloc[0].start_idx), int(prior.iloc[0].end_idx),
                            provenance=provenance, prior=prior)


@pytest.mark.parametrize("recipe", [None, regions()])
def test_real_event_and_training_exports_preserve_reference_contract_without_new_numeric_columns(
    context_factory, tmp_path, recipe,
):
    from nanoscope.analysis.scoring import write_events_export, write_training_export

    result = run(context_factory, DetectSchmitt, signal(), recipe)
    provenance = result.payload.provenance
    optional = (KEY, "baseline_reference_applied", "baseline_reference_details",
                "baseline_estimator_bypassed", "baseline_polarity_role")
    for writer, name, block in ((write_events_export, "events", "provenance"),
                               (write_training_export, "training", "detrend_contract")):
        paths = writer(result.payload.df, tmp_path / (name + ".csv"),
                       provenance=provenance, source_file="reference-integration.h5",
                       detector=result.module_id)
        metadata = json.loads(paths["meta"].read_text())
        if recipe is None:
            assert all(key not in metadata[block] for key in optional)
        else:
            for key in (*optional, "baseline_param_source"):
                assert metadata[block][key] == provenance[key]
            assert metadata[block][KEY] == recipe
        exported = pd.read_csv(paths["data"], float_precision="round_trip")
        assert not any("open_pore_reference" in column for column in exported.columns)
        if name == "events":
            assert list(exported.columns) == list(result.payload.df.columns)
            for column in MEASUREMENTS[:-1]:
                np.testing.assert_array_equal(exported[column], result.payload.df[column])
