"""Beta sweep regressions for cancellation, QC semantics and malformed inputs."""
import threading

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import (
    BaselineConfig, UNPROBED_DEFAULT_WARNING, baseline_progress_scope,
    clear_baseline_cache,
)
from nanoscope.analysis.cancel import OperationCancelled, cancel_scope, current_cancel
from nanoscope.analysis.characterize import characterize, recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.measurement_quality import validated_measurement_quality
from nanoscope.analysis.merge import merge_by_gap, merge_region
from nanoscope.analysis.open_pore_reference import OpenPoreReference


DETECTORS = [detect_events, detect_events_cusum]
FS = 50_000.0


def _clean_trace(n=100_000):
    x = np.random.default_rng(11).normal(-500.0, 5.0, n).astype(np.float32)
    for s in np.linspace(2000, n - 2000, 20, dtype=int):
        x[s:s + 50] += 100
    return x


@pytest.mark.parametrize("detector", DETECTORS)
def test_explicit_cancellation_reaches_nested_baseline_before_next_stage(detector):
    clear_baseline_cache()
    token = threading.Event()
    progress = []

    def request_cancel(fraction, message):
        progress.append(message)
        token.set()

    with baseline_progress_scope(request_cancel), pytest.raises(OperationCancelled):
        detector(_clean_trace(), FS, polarity=1, cancel=token)
    assert len(progress) == 1, "baseline continued publishing stages after cancellation"
    assert current_cancel() is None


@pytest.mark.parametrize("detector", DETECTORS)
def test_explicit_live_token_overrides_cancelled_outer_scope_and_restores_it(detector):
    clear_baseline_cache()
    outer = lambda: True
    with cancel_scope(outer):
        df, _ = detector(_clean_trace(), FS, polarity=1, cancel=lambda: False)
        assert not df.empty
        assert current_cancel() is outer
    assert current_cancel() is None


@pytest.mark.parametrize("detector", DETECTORS)
@pytest.mark.parametrize("name", ["min_dwell_s", "max_dwell_s", "merge_gap_s"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_detector_refuses_nonfinite_timing_before_measurement(detector, name, value):
    with pytest.raises(ValueError, match=name):
        detector(_clean_trace(5000), FS, baseline=-500, **{name: value})


@pytest.mark.parametrize("name", ["k_in", "k_out"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_schmitt_refuses_nonfinite_thresholds(name, value):
    with pytest.raises(ValueError, match=name):
        detect_events(_clean_trace(5000), FS, baseline=-500, **{name: value})


@pytest.mark.parametrize("rate", [0.0, -1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("operation", ["schmitt", "cusum", "basis", "characterize", "recharacterize"])
def test_measurement_entrypoints_refuse_invalid_rate_even_with_provided_baseline(rate, operation):
    x = _clean_trace(5000)
    with pytest.raises(ValueError, match="fs must be positive and finite"):
        if operation in ("schmitt", "cusum"):
            DETECTORS[operation == "cusum"](x, rate, baseline=-500)
        elif operation == "basis":
            from nanoscope.analysis.detect import detection_basis
            detection_basis(x, rate, baseline=-500)
        elif operation == "characterize":
            characterize(x, np.full(len(x), -500), [2000], [2050], rate, 1, noise_rms=5)
        else:
            recharacterize_span(x, rate, 2000, 2050, baseline=-500, polarity=1)


@pytest.mark.parametrize("detector", DETECTORS)
def test_clean_default_notices_remain_visible_without_blanket_review_flags(detector):
    x = _clean_trace(1_000_000)
    df, prov = detector(x, FS, polarity=1, min_dwell_s=2e-5)
    report = prov["measurement_quality"]
    assert len(df) == 20
    assert report["status"] == "no_issue_detected"
    assert all(row["status"] == "no_issue_detected" for row in report["events"])
    assert all(not row["quality_flags"] for row in report["events"])
    assert UNPROBED_DEFAULT_WARNING in report["messages"]
    assert UNPROBED_DEFAULT_WARNING in prov["baseline_warnings"]
    # Classifying the same measurement as user-chosen cannot change the numbers.
    control, _ = detector(x, FS, polarity=1, min_dwell_s=2e-5, baseline_param_source="user")
    pd.testing.assert_frame_equal(df, control, check_exact=True)
    # Old blanket classifications must not remain "current" after this update.
    report["version"] = 1
    assert validated_measurement_quality(df, prov)["state"] == "stale"


@pytest.mark.parametrize("detector", DETECTORS)
def test_known_reference_assumptions_are_unassessed_not_observed_defects(detector):
    config = BaselineConfig(open_pore_reference=OpenPoreReference(mode="constant", current_pA=-500))
    df, prov = detector(_clean_trace(), FS, polarity=1, baseline_config=config)
    report = prov["measurement_quality"]
    assert not df.empty
    assert report["status"] == "not_assessed"
    assert all(row["status"] == "not_assessed" for row in report["events"])
    assert "baseline_estimator_advisory" not in report["recording_flags"]
    assert prov["baseline_informational_warnings"]
    assert set(prov["baseline_warnings"]).issubset(report["messages"])


@pytest.mark.parametrize("bounds", [(300, 250), (300, 300), (-1, 20), (300, 5001)])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("operation", ["gap", "region"])
def test_merge_refuses_corrupt_rows_before_noop_or_absorption(bounds, single, operation):
    x = _clean_trace(5000)
    df = characterize(x, np.full(len(x), -500), [300, 600], [350, 650], FS, 1, noise_rms=5)
    df.loc[0, ["start_idx", "end_idx"]] = bounds
    if single:
        df = df.iloc[:1].copy()
    events = EventSet(df, {})
    before = events.df.copy(deep=True)
    with pytest.raises(ValueError, match="invalid span"):
        if operation == "gap":
            merge_by_gap(events, x, FS, max_gap_s=0.01)
        else:
            merge_region(events, x, FS, t0_s=0, t1_s=len(x) / FS)
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)


@pytest.mark.parametrize("module_name", ["DetectSchmitt", "DetectCusum"])
@pytest.mark.parametrize("explicit", [False, True])
def test_omitted_notch_lines_are_preserved_and_disclosed_in_result(context_factory, module_name, explicit):
    from nanoscope.analysis import modules
    params = dict(notch_enabled=True, notch_freq_hz=200, notch_harmonics=3)
    requested = [200.0, 400.0, 600.0]
    if explicit:
        requested = [60.00025, 500.0, 600.0]
        params["notch_freqs"] = requested
    with pytest.warns(RuntimeWarning, match="Nyquist"):
        result = getattr(modules, module_name)().run(
            context_factory(_clean_trace(20_000), fs=1000, params=params)
        )
    prov = result.payload.provenance
    assert prov["notch_requested_freqs"] == requested
    assert prov["notch_freqs"] == [f for f in requested if f < 500]
    assert prov["notch_omitted_freqs"] == [f for f in requested if f >= 500]
    assert any("were omitted" in warning and "Applied notch lines" in warning
               for warning in result.warnings)
    # A replay reconstructs only the lines that actually touched the measurement.
    assert modules.preprocess_params_from_provenance(prov)["notch_freqs"] == prov["notch_freqs"]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("own_note", [False, True])
def test_saturation_caveat_is_not_a_hand_label_or_automatic_rejection(accepted, own_note):
    from nanoscope.core.history import _summarise
    from nanoscope.analysis.events import DERIVED_FLAGS, FLAG_SATURATED, is_automatic_rejection
    x = _clean_trace(5000)
    df = characterize(x, np.full(len(x), -500), [300], [350], FS, 1, noise_rms=5)
    df.loc[0, "flag"] = "saturated;molecule=apo" if own_note else "saturated"
    df.loc[0, "accepted"] = accepted
    summary = _summarise(df)
    assert FLAG_SATURATED in DERIVED_FLAGS
    assert not is_automatic_rejection("saturated")
    assert summary["n_notes"] == int(own_note)
    assert summary["n_hand_rejected"] == int(not accepted)


@pytest.mark.parametrize("detector", DETECTORS)
def test_zero_reference_stays_supported_with_explicit_undefined_ratio_advisory(detector):
    config = BaselineConfig(open_pore_reference=OpenPoreReference(mode="constant", current_pA=0))
    df, prov = detector(_clean_trace(), FS, polarity=-1, baseline_config=config)
    assert not df.empty
    assert (df.baseline_pA == 0).all()
    assert df.frac_depth.isna().all()
    assert any("zero current" in w and "undefined" in w for w in prov["baseline_warnings"])
    assert any("zero current" in w for w in prov["measurement_quality"]["messages"])
    assert prov["measurement_quality"]["status"] == "review_required"


@pytest.mark.parametrize("reference", [1e38, 1e300])
@pytest.mark.parametrize("operation", ["schmitt", "cusum", "basis"])
def test_unresolvable_noise_names_supplied_reference_and_precision(reference, operation):
    from nanoscope.analysis.detect import detection_basis
    x = _clean_trace()
    assert np.std(x) > 1  # The recording itself plainly has measurable variation.
    config = BaselineConfig(open_pore_reference=OpenPoreReference(mode="constant", current_pA=reference))
    entrypoint = detection_basis if operation == "basis" else DETECTORS[operation == "cusum"]
    with pytest.raises(ValueError, match="supplied open-pore reference/baseline.*cannot resolve detection noise") as caught:
        entrypoint(x, FS, polarity=1, baseline_config=config)
    assert "numeric precision" in str(caught.value)
    assert "trace with no measurable noise" not in str(caught.value)
