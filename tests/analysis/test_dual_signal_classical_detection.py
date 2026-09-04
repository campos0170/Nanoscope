"""Regression contract for classical high-pass candidate localization.

High-pass may help a detector find boundaries, but a zero-DC ringing trace cannot
carry the open-pore baseline or physical event amplitudes.  These tests pin the two
domains independently, including the seed-0 fixture where the high-pass trace votes
for the opposite polarity from the recording.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import (
    baseline_config_from_params,
    characterisation_baseline,
)
from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.filters import lowpass, noise_rms
from nanoscope.analysis.modules import (
    DetectCusum,
    DetectSchmitt,
    characterization_params,
    depth_floor_from_provenance,
    detection_signal,
    detection_signal_key,
    highpass_measurement_warning,
    preprocess_for_detection,
)
from nanoscope.analysis.scoring import training_provenance

FS = 100_000.0
EVENTS = (
    (10_000, 10_300),
    (30_000, 30_250),
    (50_000, 50_400),
    (70_000, 70_200),
    (85_000, 85_350),
)


def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.full(100_000, 100.0) + rng.normal(0.0, 1.5, 100_000)
    for start, stop in EVENTS:
        raw[start:stop] -= 30.0
    return np.ascontiguousarray(raw, dtype=np.float32)


CASES = (
    (
        DetectSchmitt,
        {"k_in": 5.0, "k_out": 2.0},
        ((10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
         (70_000, 70_200), (85_000, 85_350)),
    ),
    (
        DetectCusum,
        {"cusum_k": 0.75, "cusum_h": 28.0},
        ((8_226, 8_876), (10_000, 10_295), (30_000, 30_250),
         (50_000, 50_399), (70_000, 70_193), (83_284, 83_846),
         (85_000, 85_350), (86_667, 87_920)),
    ),
)


@pytest.mark.parametrize(
    "module_cls,core,detector_params",
    (
        (DetectSchmitt, detect_events, {"k_in": 5.0, "k_out": 2.0}),
        (DetectCusum, detect_events_cusum,
         {"cusum_k": 0.75, "cusum_h": 28.0}),
    ),
)
def test_no_highpass_adapter_rows_remain_identical_to_the_core(
    context_factory, module_cls, core, detector_params
):
    """The ordinary one-signal path must not pay a numerical compatibility cost."""
    raw = _trace()
    params = {**detector_params, "min_dwell_s": 5e-4}
    result = module_cls().run(context_factory(raw, fs=FS, params=params))
    cfg, source = baseline_config_from_params(params)
    direct, _ = core(
        preprocess_for_detection(raw, FS, params),
        FS,
        units="pA",
        recording="synthetic.dat",
        pore_id="",
        baseline_config=cfg,
        baseline_param_source=source,
        **params,
    )
    pd.testing.assert_frame_equal(
        result.payload.df[list(direct.columns)].reset_index(drop=True),
        direct.reset_index(drop=True),
        check_exact=True,
    )


@pytest.mark.parametrize("module_cls,detector_params,expected_bounds", CASES)
def test_auto_polarity_comes_from_measurement_branch(
    context_factory, module_cls, detector_params, expected_bounds
):
    params = {
        **detector_params,
        "min_dwell_s": 5e-4,
        "hp_enabled": True,
        "hp_cutoff_hz": 40.0,
    }
    result = module_cls().run(context_factory(_trace(), fs=FS, params=params))
    prov = result.payload.provenance
    got = tuple(zip(result.payload.df["start_idx"], result.payload.df["end_idx"]))

    assert got == expected_bounds
    assert prov["polarity_requested"] is None
    assert prov["polarity"] == -1
    assert prov["polarity_source"] == "measured"
    assert prov["detection_polarity"] == -1
    # The unconstrained high-pass vote is the known trap; it remains auditable but
    # is not allowed to choose the applied direction.
    assert prov["detection_polarity_measured"] == +1
    assert prov["candidate_signal_polarity_measured"] == +1
    assert prov["polarity_domain_disagreement"] is True
    assert prov["polarity_retry_after_highpass_mismatch"] is True
    assert prov["candidate_localization_attempts"] == 2


@pytest.mark.parametrize("module_cls,detector_params,_bounds", CASES)
def test_candidate_bounds_and_measurement_metrics_are_independent(
    context_factory, module_cls, detector_params, _bounds
):
    raw = _trace()
    params = {
        **detector_params,
        "min_dwell_s": 5e-4,
        "hp_enabled": True,
        "hp_cutoff_hz": 40.0,
    }
    result = module_cls().run(context_factory(raw, fs=FS, params=params))
    frame, prov = result.payload.df, result.payload.provenance

    measurement = preprocess_for_detection(
        raw, FS, characterization_params(params))
    cfg, source = baseline_config_from_params(params)
    baseline_result = characterisation_baseline(
        measurement, FS, config=cfg, polarity=-1, param_source=source)
    baseline = np.asarray(baseline_result.baseline, dtype=np.float64)
    measurement64 = np.asarray(measurement, dtype=np.float64)
    measurement_noise = float(noise_rms(measurement64 - baseline))
    expected = characterize(
        measurement64,
        baseline,
        frame["start_idx"].to_numpy(dtype=np.int64),
        frame["end_idx"].to_numpy(dtype=np.int64),
        FS,
        -1,
        noise_rms=measurement_noise,
        filter_rise_s=prov.get("filter_rise_s"),
        max_dwell_s=prov.get("max_dwell_s"),
        clip_bounds=(0, raw.size),
        pore_id="",
        recording="synthetic.dat",
        units="pA",
    )

    quantitative = (
        "baseline_pA", "depth_pA", "frac_depth", "ecd_pA_s",
        "max_dev_pA", "mean_dev_pA", "rise_s", "fall_s", "snr",
        "filter_limited", "n_levels",
    )
    for column in quantitative:
        if expected[column].dtype.kind in "biu":
            np.testing.assert_array_equal(frame[column], expected[column])
        else:
            np.testing.assert_allclose(
                frame[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )

    assert prov["characterization_hp_omitted"] is True
    assert prov["highpass_localization_only"] is True
    assert prov["noise_rms_pA"] == pytest.approx(measurement_noise)
    np.testing.assert_allclose(
        frame["snr"], frame["depth_pA"] / prov["noise_rms_pA"])
    assert prov["baseline_median_pA"] == pytest.approx(99.95198822021484)
    assert prov["detection_baseline_median_pA"] == pytest.approx(
        0.104190144687891)
    assert prov["detection_noise_rms_pA"] == pytest.approx(
        1.725675600376725)
    assert prov["measurement_noise_rms_pA"] == pytest.approx(
        1.5329976768493652)

    if module_cls is DetectSchmitt:
        assert prov["threshold_in_pA"] == pytest.approx(
            detector_params["k_in"] * prov["detection_noise_rms_pA"])
        assert prov["threshold_out_pA"] == pytest.approx(
            detector_params["k_out"] * prov["detection_noise_rms_pA"])
        assert prov["threshold_noise_source"] == "detection_noise_rms_pA"
    else:
        assert "detection_noise_rms_pA" in prov["cusum_sigma_source"]


def test_lowpass_is_applied_once_and_new_replay_omits_highpass(context_factory):
    raw = _trace()
    params = {
        "k_in": 5.0,
        "k_out": 2.0,
        "min_dwell_s": 5e-4,
        "hp_enabled": True,
        "hp_cutoff_hz": 1.0,
        "filter_cutoff_hz": 10_000.0,
        "filter_order": 8,
        "filter_kind": "bessel",
    }
    result = DetectSchmitt().run(
        context_factory(raw, fs=FS, params=params))
    prov = result.payload.provenance

    expected = lowpass(
        preprocess_for_detection(raw, FS, characterization_params(params)),
        FS,
        10_000.0,
        order=8,
        kind="bessel",
    )
    rebuilt = detection_signal(raw, FS, prov)
    double_filtered = lowpass(
        expected, FS, 10_000.0, order=8, kind="bessel")
    np.testing.assert_array_equal(rebuilt, expected)
    assert not np.array_equal(rebuilt, double_filtered)

    # HP design changes do not invalidate a cached measurement-domain trace.
    changed_hp = dict(prov, hp_cutoff_hz=250.0, hp_order=2, hp_kind="bessel")
    assert detection_signal_key(changed_hp) == detection_signal_key(prov)

    # A legacy result without either split marker must replay its historical HP
    # measurement and include the HP design in the cache key.
    legacy = dict(prov)
    legacy.pop("highpass_localization_only")
    legacy.pop("characterization_hp_omitted")
    legacy_rebuilt = detection_signal(raw, FS, legacy)
    legacy_expected = lowpass(
        preprocess_for_detection(raw, FS, params),
        FS,
        10_000.0,
        order=8,
        kind="bessel",
    )
    np.testing.assert_array_equal(legacy_rebuilt, legacy_expected)
    assert not np.array_equal(legacy_rebuilt, rebuilt)
    assert detection_signal_key(dict(legacy, hp_cutoff_hz=250.0)) != \
        detection_signal_key(legacy)


def test_split_highpass_has_no_physical_depth_floor():
    split = {
        "threshold_out_pA": 3.5,
        "highpass_localization_only": True,
        "characterization_hp_omitted": True,
    }
    assert depth_floor_from_provenance(split, column="depth_pA") is None
    assert depth_floor_from_provenance(
        split, column="frac_depth", baseline_pA=[100.0, 110.0]) is None


def test_training_sidecar_keeps_both_signal_domains(context_factory):
    result = DetectSchmitt().run(context_factory(
        _trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "hp_enabled": True, "hp_cutoff_hz": 40.0,
                "hp_order": 4, "hp_kind": "butter"}))
    meta = training_provenance(
        result.payload.provenance,
        source_file="synthetic.dat",
        detector=result.module_id,
    )
    contract = meta["detrend_contract"]
    assert contract["highpass_localization_only"] is True
    assert contract["characterization_hp_omitted"] is True
    assert contract["hp_cutoff_hz"] == 40.0
    assert contract["hp_order"] == 4
    assert contract["hp_kind"] == "butter"
    assert contract["noise_rms_pA"] == pytest.approx(
        result.payload.provenance["noise_rms_pA"])
    assert contract["detection_noise_rms_pA"] == pytest.approx(
        result.payload.provenance["detection_noise_rms_pA"])
    assert contract["detection_baseline_median_pA"] == pytest.approx(
        result.payload.provenance["detection_baseline_median_pA"])
    assert contract["polarity_domain_disagreement"] is True


@pytest.mark.parametrize("module_cls,detector_params,_bounds", CASES)
def test_highpass_warning_describes_fixed_and_remaining_limits(
    context_factory, module_cls, detector_params, _bounds
):
    result = module_cls().run(context_factory(
        _trace(), fs=FS,
        params={**detector_params, "min_dwell_s": 5e-4,
                "hp_enabled": True, "hp_cutoff_hz": 40.0}))
    warnings = [text for text in result.warnings if "High-pass was ON" in text]
    assert len(warnings) == 1
    text = warnings[0]
    assert "CANDIDATE LOCALIZATION ONLY" in text
    assert "High-pass omitted" in text
    assert "start/end, dwell, rise/fall" in text
    assert "ECD's integration limits" in text
    assert "candidate locations only" not in text
    assert "opposite" in text and "repeated localization" in text


def test_legacy_highpass_warning_keeps_candidate_only_semantics():
    text = highpass_measurement_warning({
        "hp_enabled": True,
        "hp_cutoff_hz": 40.0,
        # Deliberately no split marker: this is a stored pre-fix result.
    })
    assert "legacy run" in text
    assert "candidate locations" in text
    assert "not reliable physical measurements" in text


def test_highpass_plus_normalize_keeps_both_independent_warnings(context_factory):
    result = DetectSchmitt().run(context_factory(
        _trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "hp_enabled": True, "hp_cutoff_hz": 1.0,
                "normalize_enabled": True, "normalize_mode": "robust"}))
    hp = [text for text in result.warnings if "High-pass was ON" in text]
    normalized = [text for text in result.warnings if "Normalize was ON" in text]
    assert len(hp) == len(normalized) == 1
    assert "Normalize" in hp[0] and "disclosed separately" in hp[0]
    assert "DIMENSIONLESS" in normalized[0]


def test_user_polarity_override_is_honored_without_claiming_a_retry(context_factory):
    result = DetectSchmitt().run(context_factory(
        _trace(), fs=FS,
        params={"k_in": 5.0, "k_out": 2.0, "min_dwell_s": 5e-4,
                "polarity": -1, "hp_enabled": True, "hp_cutoff_hz": 40.0}))
    prov = result.payload.provenance
    assert prov["polarity_requested"] == -1
    assert prov["polarity_source"] == "provided"
    assert prov["detection_polarity"] == -1
    assert prov["detection_polarity_measured"] == +1
    assert prov["candidate_localization_attempts"] == 1
    assert prov["polarity_retry_after_highpass_mismatch"] is False
    text = next(w for w in result.warnings if "High-pass was ON" in w)
    assert "honored the user's direction" in text
    assert "repeated localization" not in text
