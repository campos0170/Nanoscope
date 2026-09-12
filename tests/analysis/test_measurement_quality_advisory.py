"""Additive QC exposes uncertainty without changing scientific results/curation."""
import copy
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.events import empty_df
from nanoscope.analysis.measurement_quality import (
    build_measurement_quality, invalidate_measurement_quality,
    report_recording_messages, validated_measurement_quality,
)
from nanoscope.analysis.open_pore_reference import OpenPoreReference


def _fixture(spans=((1000, 2000),), eaten=150.0):
    fs = 10_000.0
    x = np.random.default_rng(319).normal(1000.0, 0.5, 5000)
    b = np.full(len(x), 1000.0)
    for s, e in spans:
        x[s:e] -= 200.0
        b[s:e] -= eaten
    df = characterize(x, b, np.array([s for s, _ in spans]),
                      np.array([e for _, e in spans]), fs, -1, noise_rms=0.5)
    prov = dict(fs=fs, baseline_method="masked_median", baseline_polarity=-1,
                polarity=-1, polarity_evidence={"verdict": "measured"})
    return x, b, df, prov


def _report(x, b, df, prov):
    return build_measurement_quality(x, b, df, prov["fs"], prov)


def test_long_event_baseline_following_is_advisory_and_preserves_every_value():
    x, b, df, prov = _fixture()
    old_df, old_x, old_b, old_prov = df.copy(deep=True), x.copy(), b.copy(), copy.deepcopy(prov)
    report = _report(x, b, df, prov)
    assert report["events"][0]["baseline_intrusion_fraction"] == pytest.approx(0.75, abs=0.003)
    assert "baseline_intrusion_suspected" in report["events"][0]["quality_flags"]
    assert report["status"] == "review_required"
    assert df.iloc[0].depth_pA < 60  # Known truth is 200; never silently corrected.
    pd.testing.assert_frame_equal(df, old_df, check_exact=True)
    np.testing.assert_array_equal(x, old_x)
    np.testing.assert_array_equal(b, old_b)
    assert prov == old_prov
    json.dumps(report, allow_nan=False)


def test_individual_intrusion_not_hidden_by_population_median():
    spans = [(500, 600), (1000, 1100), (1500, 1700), (2200, 2300), (2800, 2900)]
    x, b, df, prov = _fixture(spans, eaten=0)
    b[1500:1700] -= 150
    report = _report(x, b, df, prov)
    assert report["n_intrusion_advisories"] == 1
    assert report["events"][2]["quality_flags"] == ["baseline_intrusion_suspected"]
    assert report["events"][0]["status"] == "no_issue_detected"
    assert "do not validate" in report["limitations"]


@pytest.mark.parametrize("spans,state", [
    (((1, 100),), "not_assessed_incomplete_flanks"),
    (((1000, 1100), (1120, 1220)), "not_assessed_contaminated_flanks"),
])
def test_no_reassuring_zero_with_incomplete_or_occupied_flanks(spans, state):
    x, b, df, prov = _fixture(spans)
    report = _report(x, b, df, prov)
    assert all(r["baseline_intrusion_state"] == state for r in report["events"])
    assert all(r["baseline_intrusion_fraction"] is None for r in report["events"])
    assert report["n_intrusion_unassessed"] == len(df)


def test_excluded_span_invalidates_flank_reference():
    x, b, df, prov = _fixture()
    prov["exclude_regions"] = [(0.098, 0.099)]
    assert _report(x, b, df, prov)["events"][0]["baseline_intrusion_state"] == "not_assessed_contaminated_flanks"


def test_zero_events_retains_polarity_disagreement_and_completeness_caveat():
    x, b, _, prov = _fixture()
    prov["polarity"] = 1
    report = _report(x, b, empty_df(), prov)
    assert "baseline_detector_polarity_disagreement" in report["recording_flags"]
    assert "no_events_measurement_quality_unassessed" in report["recording_flags"]
    assert not report["events"]
    assert any("zero count" in s for s in report["messages"])


def test_missing_polarity_is_not_certification():
    x, b, df, prov = _fixture()
    prov.pop("baseline_polarity")
    prov["polarity_evidence"] = {"verdict": "undetermined"}
    report = _report(x, b, df, prov)
    assert "baseline_polarity_not_assessed" in report["recording_flags"]
    assert "detector_polarity_undetermined" in report["recording_flags"]


@pytest.mark.parametrize("mode", ["constant", "intervals"])
def test_explicit_reference_never_compared_with_unused_baseline_polarity(mode):
    x, b, df, prov = _fixture()
    prov.update(baseline_reference_applied=True, baseline_polarity=0,
                baseline_open_pore_reference={"mode": mode})
    report = _report(x, b, df, prov)
    assert report["reference_mode"] == mode
    assert report["baseline_detector_polarity_state"] == "not_applicable_user_reference"
    assert "baseline_detector_polarity_disagreement" not in report["recording_flags"]
    assert report["events"][0]["baseline_intrusion_fraction"] is None
    assert report["events"][0]["status"] == "not_assessed"
    assert any("not certified" in s for s in report["messages"])


def test_polarity_independent_estimator_is_not_false_contradiction():
    x, b, df, prov = _fixture()
    prov.update(baseline_method="constant", baseline_polarity=1)
    report = _report(x, b, df, prov)
    assert report["baseline_detector_polarity_state"] == "not_applicable_polarity_independent_baseline"
    assert "baseline_detector_polarity_disagreement" not in report["recording_flags"]


@pytest.mark.parametrize("edit", ["resize", "depth", "reference", "subset", "amplitude", "explicit"])
def test_changed_measurement_cannot_reuse_current_advisory(edit):
    x, b, df, prov = _fixture()
    prov["measurement_quality"] = _report(x, b, df, prov)
    assert validated_measurement_quality(df, prov)["state"] == "current"
    if edit == "resize":
        df.loc[0, "end_idx"] += 1
    elif edit == "depth":
        df.loc[0, "depth_pA"] += 1
    elif edit == "reference":
        prov["baseline_polarity"] = 1
    elif edit == "subset":
        df = df.iloc[0:0].copy()
    elif edit == "amplitude":
        df["amplitude_start_idx"] = pd.Series([pd.NA], dtype="Int64")
    else:
        invalidate_measurement_quality(prov, "source current was replaced")
    report = validated_measurement_quality(df, prov)
    assert report["state"] == "stale"
    assert report["status"] == "not_assessed"
    assert report["events"] == []
    assert report["previous_report"]["events"]
    assert any("stale" in s for s in report_recording_messages(df, prov))


def test_curation_and_json_roundtrip_do_not_remeasure_or_certify():
    x, b, df, prov = _fixture()
    prov["measurement_quality"] = _report(x, b, df, prov)
    df.loc[0, "accepted"] = False
    df.loc[0, "flag"] = "reviewed=scientist"
    restored = json.loads(json.dumps(prov, allow_nan=False))
    report = validated_measurement_quality(df, restored)
    assert report["state"] == "current"
    assert report["events"][0]["status"] == "review_required"
    assert report["status"] != "qualified"


def test_missing_report_has_explicit_details_message():
    assert validated_measurement_quality(empty_df(), {})["state"] == "missing"
    assert "not been assessed" in report_recording_messages(empty_df(), {})[0]


def test_persisted_stale_report_does_not_nest_or_recopy_on_each_details_read():
    x, b, df, prov = _fixture()
    prov["measurement_quality"] = _report(x, b, df, prov)
    df.loc[0, "depth_pA"] += 1
    prov["measurement_quality"] = validated_measurement_quality(df, prov)
    first = prov["measurement_quality"]
    second = validated_measurement_quality(df, prov, copy_report=False)
    assert second is first
    assert second["previous_report"]["state"] == "current"
    assert "previous_report" not in second["previous_report"]


def test_unassessed_only_status_is_not_review_required():
    x, b, _, prov = _fixture()
    prov.pop("baseline_polarity")
    prov["polarity_evidence"] = {"verdict": "undetermined"}
    report = _report(x, b, empty_df(), prov)
    assert report["status"] == "not_assessed"


@pytest.mark.parametrize("module_name,function_name", [
    ("detect", "detect_events"), ("detect_cusum", "detect_events_cusum"),
])
@pytest.mark.parametrize("mode", ["automatic", "known_current", "intervals", "zero"])
def test_detector_numeric_and_acceptance_results_identical_with_and_without_qc(monkeypatch, module_name, function_name, mode):
    module = importlib.import_module(f"nanoscope.analysis.{module_name}")
    detect = getattr(module, function_name)
    fs = 10_000.0
    x = np.random.default_rng(612).normal(120, 0.5, 20_000)
    # Long, majority-blocked challenge: a known reference is not assumed by Automatic.
    for s, e in [(2000, 7000), (8000, 13000), (14000, 19000)]:
        x[s:e] -= 60
    kwargs = dict(polarity=-1, min_dwell_s=0.001)
    if mode == "known_current":
        kwargs["baseline_config"] = BaselineConfig(open_pore_reference=OpenPoreReference(mode="constant", current_pA=120))
    elif mode == "intervals":
        kwargs["baseline_config"] = BaselineConfig(open_pore_reference=OpenPoreReference(mode="intervals", intervals_s=((0.0, 0.1),)))
    elif mode == "zero":
        kwargs["min_dwell_s"] = 10
    before = x.copy()
    actual_df, actual_prov = detect(x, fs, **kwargs)
    assert "measurement_quality" in actual_prov
    assert validated_measurement_quality(actual_df, actual_prov)["state"] == "current"
    if mode == "zero":
        assert actual_df.empty
        assert "no_events_measurement_quality_unassessed" in actual_prov["measurement_quality"]["recording_flags"]
    elif mode in ("known_current", "intervals"):
        assert actual_prov["measurement_quality"]["baseline_detector_polarity_state"] == "not_applicable_user_reference"
    else:
        assert actual_prov["measurement_quality"]["status"] == "review_required"
        assert "baseline_estimator_advisory" in actual_prov["measurement_quality"]["recording_flags"]
    monkeypatch.setattr(module, "build_measurement_quality", lambda *args, **kwargs: None)
    expected_df, expected_prov = detect(x, fs, **kwargs)
    pd.testing.assert_frame_equal(actual_df, expected_df, check_exact=True)
    np.testing.assert_array_equal(x, before)
    for prov in (actual_prov, expected_prov):
        prov.pop("measurement_quality")
        prov.pop("detected_at")
    assert actual_prov == expected_prov


@pytest.mark.parametrize("module_name", ["DetectSchmitt", "DetectCusum"])
@pytest.mark.parametrize("hp", [False, True])
def test_module_adapters_report_current_measurement_domain(context_factory, module_name, hp):
    from nanoscope.analysis import modules
    fs = 20_000.0
    x = np.random.default_rng(923).normal(100, 0.5, 40_000).astype(np.float32)
    for s, e in [(4000, 4200), (12000, 12500), (21000, 21200), (30000, 30100)]:
        x[s:e] -= 30
    params = dict(polarity=-1, min_dwell_s=0.0002,
                  filter_cutoff_hz=3000.0, hp_enabled=hp, hp_cutoff_hz=50.0)
    result = getattr(modules, module_name)().run(context_factory(x, fs=fs, params=params))
    events = result.payload
    prov = events.provenance
    report = validated_measurement_quality(events.df, prov)
    assert report["state"] == "current"
    assert report["n_events"] == len(events.df)
    assert bool(prov.get("highpass_localization_only")) == hp
    if not events.df.empty:
        events.df.loc[events.df.index[0], "depth_pA"] += 1
        assert events.provenance["measurement_quality"]["state"] == "stale"
        assert validated_measurement_quality(events.df, events.provenance)["state"] == "stale"
